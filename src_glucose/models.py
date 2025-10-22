from collections import deque
from contextlib import nullcontext
from typing import Tuple, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Normal, TransformedDistribution, TanhTransform, AffineTransform
from tqdm import tqdm
from gym_wrappers import TOTAL_SIZE


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class MiniGridEncoder(nn.Module):
    """
    A feature extractor that uses self-attention on the input features.

    This network treats the input features as a sequence and uses a
    Multi-Head Attention block to learn the relationships between them.

    :param observation_shape: The shape of the input observation.
    :param hidden_dim: The final output dimension of the extractor.
    :param embed_dim: The dimension to embed each input feature into.
    :param num_heads: The number of attention heads. Must be a divisor of embed_dim.
    """

    def __init__(
            self,
            observation_shape,
            hidden_dim: int = 128,
            embed_dim: int = 64,
            num_heads: int = 4,
    ):
        super().__init__()

        # Ensure that embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        num_inputs = observation_shape[0]  # Should be 6

        # 1. Input Embedding Layer
        # This layer projects each of the 6 input features into a richer,
        # higher-dimensional space (the embed_dim).
        self.embed_layer = nn.Linear(num_inputs, embed_dim)

        # We will treat the 6 embedded features as a sequence of length 6

        # 2. Multi-Head Self-Attention Layer
        # This layer allows the 6 embedded features to interact and weigh their
        # importance relative to each other.
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 3. Final Feed-Forward Network (MLP)
        # This processes the context-aware output from the attention layer
        # to produce the final feature vector.
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
        )

        # Use your original weight initialization method
        self.init_weights()

    def init_weights(self):
        """Initializes weights for the network."""

        def _init_fn(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming x has shape (batch_size, 6)

        # 1. Embed the input
        # Input: (batch, 6) -> Output: (batch, embed_dim)
        embedded = self.embed_layer(x)

        # 2. Prepare for attention by creating a "sequence" of length 1
        # The attention layer expects a sequence. We can treat our single
        # embedded vector as a sequence with one item.
        # Shape: (batch, 1, embed_dim)
        seq = embedded.unsqueeze(1)

        # 3. Apply self-attention
        # Query, Key, and Value are all the same for self-attention.
        # attn_output shape: (batch, 1, embed_dim)
        attn_output, _ = self.attention(query=seq, key=seq, value=seq)

        # 4. Remove the sequence dimension
        # Shape: (batch, embed_dim)
        processed_features = attn_output.squeeze(1)

        # 5. Pass through the final MLP
        # Shape: (batch, hidden_dim)
        return self.fc(processed_features)


class PPOMiniGridEncoder(MiniGridEncoder):
    def __init__(self, observation_shape: Tuple[int,], hidden_dim: int = 128,
                 *args, **kwargs) -> None:
        # Ignore all zeros at the start
        self.shrink_obs = lambda x: x
        super().__init__(observation_shape, hidden_dim)


class OfflineMiniGridEncoder(MiniGridEncoder):
    def __init__(self, observation_shape: Tuple[int,], hidden_dim: int = 128,
                 input_scaling: bool = True, *args, **kwargs) -> None:
        # additionally ignore flag channel at the start
        self.shrink_obs = lambda x: x # x[:, 1:]
        # new_obs_shape = (observation_shape[0] - 1,)

        # super().__init__(new_obs_shape, hidden_dim)
        super().__init__(observation_shape, hidden_dim)


class PPOMiniGridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,
                 features_dim: int = 512) -> None:  # , normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        self.net = PPOMiniGridEncoder(observation_space.shape, hidden_dim=features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class ScaleAction(nn.Module):
    def __init__(self, low: float, high: float):
        super().__init__()
        self.register_buffer('low', torch.tensor(low, dtype=torch.float32))
        self.register_buffer('high', torch.tensor(high, dtype=torch.float32))
        self.device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale from [0, 1] to [low, high]
        if self.device is None or self.device != x.device:
            self.device = x.device
            self.register_buffer('low', self.low.to(self.device))
            self.register_buffer('high', self.high.to(self.device))
        return x * (self.high - self.low) + self.low


class CustomNet(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], output_size: int, hidden_dim: int = 128,
                 device: str = 'cpu', action_encoder=None, feature_extractor=None, dropout_p: float = 0.0,
                 *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        if feature_extractor is None:
            self.encoder = OfflineMiniGridEncoder(observation_shape=observation_shape,
                                                  hidden_dim=hidden_dim,
                                                  dropout_p=dropout_p).to(device)
        else:
            self.encoder = feature_extractor

        self.action_encoder = action_encoder
        self._has_actions = action_encoder is not None

        decoder_input_size = hidden_dim * (1 + int(self._has_actions))

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_input_size // 2),
            nn.LayerNorm(decoder_input_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(decoder_input_size // 2, decoder_input_size // 2),
            nn.LayerNorm(decoder_input_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(decoder_input_size // 2, output_size)
        ).to(device)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for Conv2d/Linear with Kaiming normal,
        biases with zeros, Norm layers with weight=1, bias=0.
        The final Linear layer in the decoder is zero-initialized.
        """

        def _init_fn(m: nn.Module):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # apply to all submodules
        self.apply(_init_fn)

        # special-case: zero-init the very last decoder layer
        if isinstance(self.decoder[-1], nn.Linear):
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x: torch.Tensor, actions: torch.Tensor = None, flags: torch.Tensor = None) -> torch.Tensor:
        x, actions, flags = self._ndarray_to_tensor(x, actions, flags)
        x = x.to(dtype=torch.float32)

        # (N, C, H, W) -> (N, hidden_dim)
        hidden = self.encoder(x)
        if self._has_actions:
            hidden = torch.concatenate([hidden,
                                        self.action_encoder(actions.view(-1, 1).to(dtype=torch.float32))], dim=-1)

        # (N, hidden_dim) -> (N, output_size)
        output = self.decoder(hidden)

        if flags is not None:
            output = output.view(x.size(0), 2, -1)
            output = torch.take_along_dim(output, flags.long().unsqueeze(-1), 1).squeeze(1)

        return output

    def _ndarray_to_tensor(self, *arrays):
        new_tensors = []
        for array in arrays:
            if array is None:
                new_tensors.append(None)
                continue
            if not isinstance(array, torch.Tensor):
                array = torch.tensor(array)
            new_tensors.append(array.to(self._device))

        return new_tensors

    def enable_mc_dropout(self):
        """Enable MC Dropout during inference."""

        def apply_mc_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        self.apply(apply_mc_dropout)


class RecurrentNet(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], output_size: int, hidden_dim: int = 128,
                 recurrent_hidden_size: int = 128, device: str = 'cpu', action_encoder=None,
                 feature_extractor=None, dropout_p: float = 0.0, burn_in_length: int = 0, *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        self.recurrent_hidden_size = recurrent_hidden_size

        if feature_extractor is None:
            # This encoder now processes individual frames
            self.encoder = OfflineMiniGridEncoder(observation_shape=observation_shape, hidden_dim=hidden_dim,
                                                  dropout_p=dropout_p).to(device)
        else:
            self.encoder = feature_extractor

        self.action_encoder = action_encoder
        self._has_actions = action_encoder is not None
        self.burn_in_length = burn_in_length

        # Input to LSTM is the feature vector (plus action embedding if present)
        lstm_input_size = hidden_dim * (1 + int(self._has_actions))

        # --- Recurrent Layer ---
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=recurrent_hidden_size,
            num_layers=1,
            batch_first=True  # Crucial for easier tensor manipulation
        ).to(device)

        # Decoder now takes the output of the LSTM
        decoder_input_size = recurrent_hidden_size
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_input_size // 2),
            nn.LayerNorm(decoder_input_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(decoder_input_size // 2, output_size)
        ).to(device)

        # Note: Weight initialization is omitted for brevity but should be the same as in your original code.
        # self.init_weights()

    @torch.compile
    def forward(self, x: torch.Tensor, actions: torch.Tensor = None,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                padding_mask: Optional[torch.Tensor] = None,
                train_mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (N, T, C, H, W), where T is sequence length
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Reshape to process each frame through the encoder
        # (N, T, C, H, W) -> (N * T, C, H, W)
        x_reshaped = x.reshape(-1, *x.shape[2:])

        # (N * T, C, H, W) -> (N * T, hidden_dim)
        hidden = self.encoder(x_reshaped.to(dtype=torch.float32))

        if self._has_actions:
            # actions shape: (N, T, 1) -> (N * T, 1)
            actions_reshaped = actions.reshape(-1, 1).to(dtype=torch.float32)
            action_embedding = self.action_encoder(actions_reshaped)
            hidden = torch.cat([hidden, action_embedding], dim=-1)

        # Reshape back to sequence format for LSTM
        # (N * T, lstm_input_size) -> (N, T, lstm_input_size)
        lstm_input = hidden.view(batch_size, seq_len, -1)

        if train_mask is not None and torch.any(~(train_mask == padding_mask)):
            # --- Burn-in Logic with per-sequence length ---
            burn_in_lengths = self._get_burn_in_lengths(train_mask)  # shape: [B]

            # Create a mask of shape (B, T) to identify training steps.
            timesteps = torch.arange(seq_len, device=lstm_input.device).expand(len(burn_in_lengths), seq_len)
            is_training_mask = timesteps >= burn_in_lengths.unsqueeze(1)

            # Use torch.where to create a new input tensor.
            # For training steps, use the original input to preserve the computation graph.
            # For burn-in steps, use a detached version to cut the graph.
            lstm_input = torch.where(
                is_training_mask.unsqueeze(-1),  # expand to [B, T, 1] to broadcast
                lstm_input,
                lstm_input.detach()
            )

        # Now, proceed with the masked input as if it were the original.
        packed_input = self._pack_sequence_maybe(lstm_input, padding_mask)
        lstm_out, next_hidden_state = self.lstm(packed_input, hidden_state)
        lstm_out = self._unpack_sequence_maybe(lstm_out, seq_len)

        # Reshape for the decoder
        # (N, T, recurrent_hidden_size) -> (N * T, recurrent_hidden_size)
        decoder_input = lstm_out.reshape(-1, self.recurrent_hidden_size)

        # (N * T, recurrent_hidden_size) -> (N * T, output_size)
        output = self.decoder(decoder_input)

        # Reshape final output back to sequence format
        # (N * T, output_size) -> (N, T, output_size)
        output = output.view(batch_size, seq_len, -1)

        return output, next_hidden_state

    @staticmethod
    def _get_burn_in_lengths(train_mask):
        """
        Calculates the length of the burn-in period for each sequence in the batch.
        A burn-in period is the sequence of leading False values in the train_mask.

        Args:
            train_mask (Tensor): A boolean tensor of shape [B, T, 1].

        Returns:
            Tensor: A 1D tensor of shape [B] with the burn-in length for each sequence.
        """
        # Squeeze to shape [B, T]
        mask = train_mask.squeeze(-1)

        # `argmax` finds the index of the first `True`. This index is the number of
        # leading `False` values, which is our burn-in length.
        # To handle sequences that are all `False` (all burn-in), we add a `True`
        # sentinel value at the end. `argmax` will then return `seq_len`.
        sentinel = torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
        mask_with_sentinel = torch.cat([mask, sentinel], dim=1)

        # Cast to int for argmax
        burn_in_lengths = torch.argmax(mask_with_sentinel.to(torch.int), dim=1)
        return burn_in_lengths

    @staticmethod
    def _pack_sequence_maybe(lstm_input, padding_mask):
        if padding_mask is not None:
            lengths = padding_mask.sum(dim=1).squeeze(-1).cpu().to(torch.int64)
            lengths = torch.clamp(lengths, min=1)  # Ensure no zero lengths
            lstm_input = pack_padded_sequence(
                lstm_input,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
        return lstm_input

    @staticmethod
    def _unpack_sequence_maybe(lstm_out, seq_len):
        if isinstance(lstm_out, PackedSequence):
            lstm_out, _ = pad_packed_sequence(
                lstm_out,
                batch_first=True,
                total_length=seq_len
            )
        return lstm_out


class _RecurrentBase(nn.Module):
    def __init__(self, observation_shape: Tuple[int, int, int], action_space: spaces.Box,
                 hidden_dim: int = 128, gamma: float = 0.99, recurrent_hidden_size: int = None, batch_size: int = 128,
                 device: str = 'cpu', decoy_interval: int = 0, critic_lr: float = 3e-4,
                 value_lr: float = 3e-4, actor_lr: float = 3e-4, sequence_length: int = 64,
                 burn_in_length: int = 20, *args, **kwargs):
        super().__init__()
        self._cloning_only = False
        self._device = device
        self.decoy_interval = decoy_interval
        self.scaler = None
        self._batch_diff = None
        self._critic_lr = critic_lr
        self._value_lr = value_lr
        self._actor_lr = actor_lr
        self._observation_shape = observation_shape
        self._action_low = action_space.low[0]
        self._action_high = 2.0 #action_space.high[0]
        self._hidden_dim = hidden_dim
        self._recurrent_hidden_size = self._hidden_dim if recurrent_hidden_size is None else recurrent_hidden_size
        self._batch_size = batch_size
        self._eps = 1e-5
        self._tanh_scale = 1.0

        # Does this need to be removed post-burnin?
        if self.decoy_interval == 0:
            sequence_length *= int((1 + TOTAL_SIZE) / 2)
        self._sequence_length = sequence_length
        self._burn_in_length = burn_in_length
        # Total sequence length including burn-in
        self._total_seq_len = self._burn_in_length + self._sequence_length

        # Preload some tensors
        self._gamma = torch.tensor(gamma, device=self._device)
        self._buffered_sequence_arange = torch.arange(self._total_seq_len, device=self._device).unsqueeze(0).expand(
            self._batch_size, self._total_seq_len)
        self._buffered_batch_arange = torch.arange(self._batch_size, device=self._device).unsqueeze(1).expand(
            self._batch_size, self._total_seq_len)
        self._buffered_empty = torch.full((self._batch_size, self._total_seq_len), self._total_seq_len,
                                          device=self._device)

        self._buffered_pow_t = torch.pow(self._gamma,
                                         torch.arange(self._total_seq_len, device=self._device,
                                                      dtype=self._gamma.dtype))

    def fit(
            self,
            dataset,
            n_epochs_train: int = 1,
            n_epochs_per_eval: int = 1,
            evaluators=None,
            show_progress: bool = True,
            experiment_name: str = None,
            decoy_interval: int = 0,
            dataset_kwargs: Optional[Dict] = None
    ):
        # Initialise our dataset and loss dictionary
        dataset_kwargs = dict() if dataset_kwargs is None else dataset_kwargs
        dataset.set_generate_params(self._device, max_sequence_length=self._sequence_length,
                                    burn_in_length=self._burn_in_length, **dataset_kwargs)
        self.decoy_interval = decoy_interval
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')

        loss_dict = self._reset_loss_dict()

        # Start training
        with tqdm(total=n_epochs_train * len(dataset), desc="Progress", mininterval=2.0,
                  disable=not show_progress) as pbar:
            for epoch in range(1, n_epochs_train + 1):
                epoch_str = f"{epoch}/{n_epochs_train}"

                for batch in dataset:

                    # Update the networks
                    if not self._cloning_only:
                        loss_dict['critic_loss'].append(self._update_critic(*batch).item())
                        loss_dict['value_loss'].append(self._update_value(*batch).item())
                    loss_dict['policy_loss'].append(self._update_actor(*batch).item())

                    # Soft update of target value network
                    self.sync_target_networks()

                    pbar.update(1)
                    pbar.set_postfix(epoch=epoch_str,
                                     policy_loss=f"{np.mean(loss_dict['policy_loss']):.5f}",
                                     critic_loss=f"{np.mean(loss_dict['critic_loss']):.5f}",
                                     value_loss=f"{np.mean(loss_dict['value_loss']):.5f}",
                                     refresh=False)

                # Logging
                if (epoch + 1) % n_epochs_per_eval == 0 and evaluators is not None:
                    loss_dict, log_dict = self._log_progress(
                        epoch=epoch,
                        loss_dict=loss_dict,
                        experiment_name=experiment_name,
                        evaluators=evaluators
                    )

        return log_dict

    def _update_critic(self, *args):
        return torch.tensor(torch.nan)

    def _update_value(self, *args):
        return torch.tensor(torch.nan)

    def _update_actor(self, *args):
        return torch.tensor(torch.nan)

    def sync_target_networks(self):
        pass

    def predict(self, obs: np.ndarray, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predicts an action for a single observation and manages the recurrent state.
        Returns the action and the next hidden state.
        """
        obs_tensor = self._to_tensors(obs)[0]
        # Add batch and sequence dimensions: (L,W) -> (N, L,W)
        obs_tensor = obs_tensor.unsqueeze(0)

        # Pass the current hidden_state to the policy network
        params, next_hidden_state = self.policy_net(obs_tensor, hidden_state=hidden_state)
        mean, log_std = torch.chunk(params, 2, dim=-1)
        # 2. Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        base_dist = Normal(mean, std)
        policy_dist = TransformedDistribution(base_dist, self._transforms)

        if deterministic:
            action = torch.tanh(base_dist.mode * self._tanh_scale) * self._scale + self._loc
        else:
            action = policy_dist.rsample()

        return action.squeeze(-1).cpu().numpy(), next_hidden_state

    @torch.compile
    def _filter_to_correct_visibles(self, q1, q2, r, v_next, dones, visible, next_visible, padding_mask, train_mask):
        # shapes: q1,q2 (N,T,1 or N,T,*), r,v_next,dones,visible,next_visible (N,T,1)
        N, T = visible.shape[:2]
        dev = q1.device
        dtype = r.dtype

        # --- NEW: Apply the mask to sanitize inputs ---
        mask = padding_mask
        mask_squeezed = padding_mask.squeeze(-1)  # Shape (N, T)

        # Zero out rewards in the padded region before cumsum
        r = r * mask.to(dtype)

        # In the padded region, treat every step as if it were 'done'.
        # This prevents the 'next_true_idx' from looking past the real sequence end.
        dones = dones.squeeze(-1).bool() | ~mask_squeezed
        dones = dones.unsqueeze(-1)

        # Ensure no visible flags are true in the padded region
        visible = visible & mask
        next_visible = next_visible & mask
        # --- END OF NEW CODE ---

        vis = visible.squeeze(-1).bool()  # (N,T)
        nvis = next_visible.squeeze(-1).bool()  # (N,T)
        dn = dones.squeeze(-1).bool()  # (N,T)

        t_idx = self._buffered_sequence_arange
        none = self._buffered_empty

        def next_true_idx(mask):  # earliest j >= t if exists else T
            cand = torch.where(mask, t_idx, none)
            rev = torch.flip(cand, dims=[1])
            suf = torch.cummin(rev, dim=1).values
            return torch.flip(suf, dims=[1])

        next_done = next_true_idx(dn)
        next_nv = next_true_idx(nvis)

        # prefer next_visible unless dones occurs earlier or at the same time
        target_idx = torch.where(next_done <= next_nv, next_done, next_nv)  # (N,T)
        # --- Use train_mask to select valid steps ---
        valid = vis & (target_idx < T) & train_mask.squeeze(-1).bool()

        # gather helpers
        b_full = self._buffered_batch_arange
        b = b_full[valid]  # (M,)
        i = t_idx[valid]  # (M,)
        j = target_idx[valid]  # (M,)

        # select current Q-values at visible timesteps
        q1_sel = q1[b, i]
        q2_sel = q2[b, i]

        # n-step reward sum over [i..j] with constant gamma
        r_flat = r.squeeze(-1)  # (N,T)
        pow_t = self._buffered_pow_t  # (T,)
        rg = r_flat * pow_t  # r_t * γ^t
        S = torch.cumsum(rg, dim=1)  # prefix sum of r_t * γ^t

        Sj = S[b, j]
        Si_1 = torch.where(i > 0, S[b, i - 1], torch.zeros_like(Sj))
        gamma_neg_i = torch.pow(self._gamma, -i.to(dtype))
        R_i_j = gamma_neg_i * (Sj - Si_1)  # Σ_{k=i..j} γ^{k-i} r_k   (M,)

        # bootstrap from s_{j+1} with γ^(j-i+1) and (1 - done_j)
        vnext_j = v_next[b, j].squeeze(-1)  # (M,)
        done_j = dn[b, j].to(dtype)  # (M,)
        expo = torch.pow(self._gamma,
                         (j - i + 1).to(dtype))  # (M,)

        q_target_sel = R_i_j.unsqueeze(-1) + (expo * (1 - done_j) * vnext_j).unsqueeze(-1)

        return q1_sel, q2_sel, q_target_sel, valid

    def _to_tensors(self, *arrays):
        new_tensors = []
        for arr in arrays:
            if not isinstance(arr, torch.Tensor):
                arr = torch.tensor(arr, dtype=torch.float32)
            new_tensors.append(arr.to(self._device))
        return new_tensors

    def _log_progress(self, epoch: int, loss_dict: dict, experiment_name: str,
                      evaluators: dict = None):

        rewards = {}
        for key in evaluators.keys():
            mean_rew, std_rew = evaluators[key](self)
            rewards[key] = (mean_rew, std_rew)

        eval_str = '\n' + '=' * 40 + f"\nEpoch {epoch}:"
        for key in rewards.keys():
            eval_str += f"\n     {key} = {rewards[key][0]:.2f} +/- {rewards[key][1]:.2f}"

        eval_str += f"\n\n     policy_loss = {np.mean(loss_dict['policy_loss']):.7f}"
        eval_str += f"\n     critic_loss = {np.mean(loss_dict['critic_loss']):.7f}"
        eval_str += f"\n     value_loss = {np.mean(loss_dict['value_loss']):.7f}\n"
        eval_str += '=' * 40 + '\n'
        print(eval_str)

        return self._reset_loss_dict(), rewards

    @staticmethod
    def _reset_loss_dict():
        return {
            'critic_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'policy_loss': deque(maxlen=100)
        }


class RecurrentIQL(_RecurrentBase):
    def __init__(
            self,
            expectile: float = 0.7,
            tau_target: float = 0.005,
            dropout_p: float = 0.0,
            beta: float = 2.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._expectile = expectile
        self._cloning_only = expectile == 0.5
        self._tau_target = tau_target
        self._has_dropout = dropout_p > 0.0
        self._beta = beta
        self._scale = torch.tensor((self._action_high - self._action_low) / 2.0, device=self._device)
        self._log_scale = torch.tensor(self._scale, device=self._device).log()
        self._loc = torch.tensor((self._action_high + self._action_low) / 2.0, device=self._device)
        self._tanh_scale = 0.2  # Scale before applying tanh
        self._log_tanh_scale = torch.tensor(self._tanh_scale, device=self._device).log()
        self._transforms = [
            AffineTransform(loc=0.0, scale=self._tanh_scale, cache_size=1),
            TanhTransform(cache_size=1),
            AffineTransform(loc=self._loc, scale=self._scale, cache_size=1)
        ]

        net_kwargs = dict(
            observation_shape=self._observation_shape,
            hidden_dim=self._hidden_dim,
            recurrent_hidden_size=self._recurrent_hidden_size,
            dropout_p=dropout_p,
            device=self._device
        )

        self.feature_extractor = OfflineMiniGridEncoder(**net_kwargs).to(self._device)
        self.action_encoder = nn.Sequential(
            ScaleAction(low=self._action_low, high=self._action_high),
            nn.Linear(1, self._hidden_dim),
            nn.LayerNorm(self._hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        ).to(self._device)

        self.critic_net1 = RecurrentNet(output_size=1, action_encoder=self.action_encoder,
                                        feature_extractor=self.feature_extractor, **net_kwargs)
        self.critic_net2 = RecurrentNet(output_size=1, action_encoder=self.action_encoder,
                                        feature_extractor=self.feature_extractor, **net_kwargs)
        self.value_net = RecurrentNet(output_size=1, feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_value_net = RecurrentNet(output_size=1, feature_extractor=self.feature_extractor, **net_kwargs)
        self.policy_net = RecurrentNet(output_size=2, feature_extractor=self.feature_extractor, **net_kwargs)

        # Give both critic nets to the critic optimizer
        self.critic_optim = torch.optim.AdamW(list(self.critic_net1.parameters()) +
                                              list(self.critic_net2.parameters()), lr=self._critic_lr)
        self.value_optim = torch.optim.AdamW(self.value_net.parameters(), lr=self._value_lr)
        self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=self._actor_lr)

        # clone the value net to a target network
        self.sync_target_networks(tau=1.0)

    def get_initial_states(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a zero-initialized hidden state tuple for the LSTM."""
        h_0 = torch.zeros(1, batch_size, self._recurrent_hidden_size, device=self._device)
        c_0 = torch.zeros(1, batch_size, self._recurrent_hidden_size, device=self._device)
        return h_0, c_0

    def forward(self, obs, acts):
        # Note that this returns both the output and the hidden states
        q1 = self.critic_net1(obs, acts)
        q2 = self.critic_net2(obs, acts)
        v = self.value_net(obs)
        logits = self.policy_net(obs)
        return (q1, q2), v, logits

    def sync_target_networks(self, tau=None):
        tau = tau or self._tau_target
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    @torch.compile
    def _update_critic(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                       train_mask):
        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            # We only need the network output, not the final hidden state for training
            q1, _ = self.critic_net1(obs, acts, padding_mask=padding_mask, train_mask=train_mask)
            q2, _ = self.critic_net2(obs, acts, padding_mask=padding_mask, train_mask=train_mask)

            with torch.no_grad():
                v_next, _ = self.target_value_net(next_obs, padding_mask=next_padding_mask)
                r = rews.float()

            if self.decoy_interval == 0:
                q1, q2, q_target, _ = self._filter_to_correct_visibles(
                    q1, q2, r, v_next, dones, visible, next_visible, padding_mask, train_mask)
            else:
                q_target = r + self._gamma * (1 - dones.float()) * v_next

            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Scale the loss and perform the backward pass
        self.critic_optim.zero_grad()
        if self.scaler:
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optim)
            self.scaler.update()
        else:
            critic_loss.backward()
            self.critic_optim.step()

        return critic_loss

    @torch.compile
    def _update_value(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask, train_mask):
        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            v, _ = self.value_net(obs, padding_mask=padding_mask, train_mask=train_mask)
            with torch.no_grad():
                q1, _ = self.critic_net1(obs, acts, padding_mask=padding_mask)
                q2, _ = self.critic_net2(obs, acts, padding_mask=padding_mask)
                q = torch.min(q1, q2)

            diff = q - v
            self._batch_diff = diff.detach()  # Shape is now (N, T, 1)
            weights = torch.absolute(self._expectile - (self._batch_diff < 0).float())
            value_loss_unmasked = (weights * (diff ** 2))  # Loss per timestep

            # --- CORRECTED MASKING LOGIC ---
            # Use train_mask (which is False for padding AND burn-in)
            final_mask = train_mask.squeeze(-1)  # Shape (N, T)

            # If the decoy_interval logic is active, also apply the 'visible' mask
            if self.decoy_interval == 0:
                final_mask = final_mask & visible.squeeze(-1)

            # Apply the final mask to the loss
            masked_loss = value_loss_unmasked.squeeze(-1) * final_mask.float()

            # Calculate the mean loss ONLY over the valid, masked timesteps
            # Add a small epsilon to prevent division by zero
            value_loss = masked_loss.sum() / (final_mask.sum() + self._eps)
            # --- END OF CORRECTION ---

        # The rest of the function remains the same
        self.value_optim.zero_grad()
        if self.scaler:
            self.scaler.scale(value_loss).backward()
            self.scaler.step(self.value_optim)
            self.scaler.update()
        else:
            value_loss.backward()
            self.value_optim.step()

        return value_loss

    @torch.compile
    def _update_actor(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask, train_mask):
        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            params, _ = self.policy_net(obs, padding_mask=padding_mask, train_mask=train_mask)
            mean, log_std = torch.chunk(params, 2, dim=-1)
            # 2. Clamp log_std for numerical stability
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = log_std.exp()

            # 3. Create the Normal distribution using the policy's outputs
            base_dist = Normal(mean, std)

            # 4. Calculate Negative Log-likelihood using unsquashed actions
            # - get the 'u' expert action in the [-1, 1] range
            u_expert = (acts - self._loc) / self._scale
            u_expert = torch.clamp(u_expert, -1 + self._eps, 1 - self._eps)
            pre_tanh_action_expert = torch.atanh(u_expert) / self._tanh_scale
            nll = -base_dist.log_prob(pre_tanh_action_expert).sum(dim=-1)

            # Add the positive log-Jacobian term
            log_det_jacobian = torch.sum(torch.log(torch.clamp(1 - u_expert ** 2, self._eps, 1.0)), dim=-1)
            log_det_jacobian += u_expert.shape[-1] * self._log_tanh_scale
            log_det_jacobian += u_expert.shape[-1] * self._log_scale

            policy_loss_unmasked = nll - log_det_jacobian

            # 5. Get the IQL weights
            weights = 1.0
            if not self._cloning_only:
                weights = self._batch_diff
                if self._has_dropout:
                    weights = torch.absolute(self._expectile - (self._batch_diff < 0).float())
                else:
                    weights = torch.clip(torch.exp(self._beta * weights), -100, 100)

                # Ensure weights have the same shape as policy_loss for broadcasting
                weights = weights.squeeze(-1)  # Shape (N, T)

            # Apply the mask
            final_mask = train_mask.squeeze(-1)
            if self.decoy_interval == 0:
                final_mask = final_mask & visible.squeeze(-1)

            masked_loss = policy_loss_unmasked * weights * final_mask.float()
            policy_loss = masked_loss.sum() / (final_mask.sum() + self._eps)

        # The rest of the function remains the same
        self.policy_optim.zero_grad()
        if self.scaler:
            self.scaler.scale(policy_loss).backward()
            self.scaler.step(self.policy_optim)
            self.scaler.update()
        else:
            policy_loss.backward()
            self.policy_optim.step()
        return policy_loss


class RecurrentCQLSAC(_RecurrentBase):
    def __init__(
            self,
            cql_alpha: float = 1.0,
            tau_target: float = 0.005,
            entropy_alpha: float = 0.2,
            dropout_p: float = 0.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._tau_target = tau_target
        self._cql_alpha = cql_alpha
        self._entropy_alpha = entropy_alpha
        self._scale = 1.0  # torch.tensor((self._action_high - self._action_low) / 2.0, device=self._device)
        self._log_scale = torch.tensor(self._scale, device=self._device).log()
        self._loc = 1.0  # torch.tensor((self._action_high + self._action_low) / 2.0, device=self._device)
        self._tanh_scale = 0.2  # Scale before applying tanh
        self._log_tanh_scale = torch.tensor(self._tanh_scale, device=self._device).log()
        self._transforms = [
            AffineTransform(loc=0.0, scale=self._tanh_scale, cache_size=1),
            TanhTransform(cache_size=1),
            AffineTransform(loc=self._loc, scale=self._scale, cache_size=1)
        ]

        net_kwargs = dict(
            observation_shape=self._observation_shape,
            hidden_dim=self._hidden_dim,
            recurrent_hidden_size=self._recurrent_hidden_size,
            dropout_p=dropout_p,
            device=self._device
        )

        self.feature_extractor = OfflineMiniGridEncoder(**net_kwargs).to(self._device)
        self.action_encoder = nn.Sequential(
            ScaleAction(low=self._action_low, high=self._action_high),
            nn.Linear(1, self._hidden_dim),
            nn.LayerNorm(self._hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        ).to(self._device)

        self.critic_net1 = RecurrentNet(output_size=1, action_encoder=self.action_encoder,
                                        feature_extractor=self.feature_extractor, **net_kwargs)
        self.critic_net2 = RecurrentNet(output_size=1, action_encoder=self.action_encoder,
                                        feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_critic_net1 = RecurrentNet(output_size=1, action_encoder=self.action_encoder,
                                               feature_extractor=self.feature_extractor, **net_kwargs)
        self.target_critic_net2 = RecurrentNet(output_size=1, action_encoder=self.action_encoder,
                                               feature_extractor=self.feature_extractor, **net_kwargs)
        self.policy_net = RecurrentNet(output_size=2, feature_extractor=self.feature_extractor, **net_kwargs)

        self.critic_optim = torch.optim.AdamW(
            list(self.critic_net1.parameters()) + list(self.critic_net2.parameters()), lr=self._critic_lr
        )
        self.policy_optim = torch.optim.AdamW(self.policy_net.parameters(), lr=self._actor_lr)

        # Initialize target networks
        self.sync_target_networks(tau=1.0)

    def forward(self, obs):
        # Note that this returns both the output and the hidden states
        q1_logits = self.critic_net1(obs)
        q2_logits = self.critic_net2(obs)
        policy_logits = self.policy_net(obs)
        return q1_logits, q2_logits, policy_logits

    #@torch.compile
    def _update_critic(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask, train_mask):
        def reshape_cql(tensor):
            return tensor.view(batch_size, cql_n_samples, seq_len, 1).permute(0, 2, 1, 3).squeeze(-1)

        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            # Inverse-map dataset actions to [-1, 1] range
            u_data = (acts - self._loc) / (self._scale + self._eps)
            u_data = torch.clamp(u_data, -1.0 + self._eps, 1.0 - self._eps)

            # --- Get Q-values for actions in the dataset ---
            # These are the full-sequence [N, T, 1] predictions
            q1_pred, _ = self.critic_net1(obs, u_data, padding_mask=padding_mask, train_mask=train_mask)
            q2_pred, _ = self.critic_net2(obs, u_data, padding_mask=padding_mask, train_mask=train_mask)

            # --- Calculate Bellman Target Components ---
            with torch.no_grad():
                params, _ = self.policy_net(next_obs, padding_mask=next_padding_mask)
                mean, log_std = torch.chunk(params, 2, dim=-1)
                # Clamp log_std for numerical stability
                log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
                std = log_std.exp()
                normal_dist = Normal(mean, std)

                # Sample next action using reparameterization trick
                next_actions_pre_tanh = normal_dist.rsample()
                next_actions = torch.tanh(next_actions_pre_tanh * self._tanh_scale)

                # If using decoy_interval == 0, persist actions over non-visible steps
                if self.decoy_interval == 0:
                    next_actions_persistent = _persist_actions(next_actions, next_visible)
                else:
                    next_actions_persistent = next_actions

                # Calculate log_prob, correcting for the tanh squash
                log_jacobian = torch.log(1 - next_actions.pow(2) + self._eps) + self._log_tanh_scale
                next_log_probs = normal_dist.log_prob(next_actions_pre_tanh) - log_jacobian
                next_log_probs = next_log_probs.sum(dim=-1, keepdim=True)

                # Get Q-values for the *next* state and *sampled next* action
                next_q1, _ = self.target_critic_net1(next_obs, next_actions_persistent, padding_mask=next_padding_mask)
                next_q2, _ = self.target_critic_net2(next_obs, next_actions_persistent, padding_mask=next_padding_mask)
                next_q = torch.min(next_q1, next_q2)

                # Target V-value (logit) = Q - alpha * log_prob
                next_v = next_q - self._entropy_alpha * next_log_probs  # Shape [N, T, 1]

                r = rews.float()  # Shape [N, T, 1]
                dones = dones.float()  # Shape [N, T, 1]

            # ---- CQL loss terms (calculated on full [N, T] grid) ----
            cql_n_samples = 10
            batch_size, seq_len, obs_dim = obs.shape

            cql_obs = obs.unsqueeze(1).repeat(1, cql_n_samples, 1, 1).view(batch_size * cql_n_samples, seq_len, obs_dim)
            cql_padding_mask = padding_mask.unsqueeze(1).repeat(1, cql_n_samples, 1, 1).view(batch_size * cql_n_samples,
                                                                                             seq_len, 1)
            cql_train_mask = train_mask.unsqueeze(1).repeat(1, cql_n_samples, 1, 1).view(batch_size * cql_n_samples,
                                                                                         seq_len, 1)

            with torch.no_grad():
                params, _ = self.policy_net(cql_obs, padding_mask=cql_padding_mask)
                mean, log_std = torch.chunk(params, 2, dim=-1)
                # 2. Clamp log_std for numerical stability
                log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
                std = log_std.exp()
                normal_dist = Normal(mean, std)

                actions_pre_tanh = normal_dist.rsample()
                cql_policy_actions = torch.tanh(actions_pre_tanh * self._tanh_scale)

                # Calculate log prob
                cql_log_jacobian = torch.log(1 - cql_policy_actions.pow(2) + self._eps) + self._log_tanh_scale
                cql_policy_log_probs = normal_dist.log_prob(actions_pre_tanh) - cql_log_jacobian
                cql_policy_log_probs = cql_policy_log_probs.sum(dim=-1, keepdim=True)

            cql_uniform_actions = torch.empty(
                batch_size * cql_n_samples, seq_len, 1, device=obs.device
            ).uniform_(-1.0, 1.0)
            cql_uniform_log_probs = -torch.log(torch.tensor(2.0, device=obs.device))

            # If using decoy_interval == 0, persist actions over non-visible steps
            if self.decoy_interval == 0:
                # We need the 'visible' mask, expanded to [N*k, T, 1]
                # 'visible' shape is [N, T, 1]
                cql_visible_mask = visible.unsqueeze(1).repeat(1, cql_n_samples, 1, 1).view(
                    batch_size * cql_n_samples, seq_len, 1
                )

                cql_policy_actions_persistent = _persist_actions(cql_policy_actions, cql_visible_mask)
                cql_uniform_actions_persistent = _persist_actions(cql_uniform_actions, cql_visible_mask)
            else:
                cql_policy_actions_persistent = cql_policy_actions
                cql_uniform_actions_persistent = cql_uniform_actions

            # --- Get Q-values for all sampled actions ---
            cql_q1_policy, _ = self.critic_net1(cql_obs, cql_policy_actions_persistent, padding_mask=cql_padding_mask,
                                                train_mask=cql_train_mask)
            cql_q2_policy, _ = self.critic_net2(cql_obs, cql_policy_actions_persistent, padding_mask=cql_padding_mask,
                                                train_mask=cql_train_mask)
            cql_q1_uniform, _ = self.critic_net1(cql_obs, cql_uniform_actions_persistent, padding_mask=cql_padding_mask,
                                                 train_mask=cql_train_mask)
            cql_q2_uniform, _ = self.critic_net2(cql_obs, cql_uniform_actions_persistent, padding_mask=cql_padding_mask,
                                                 train_mask=cql_train_mask)

            # --- Apply log-prob correction (log Q - log pi) ---
            cql_q1_policy = cql_q1_policy - cql_policy_log_probs
            cql_q2_policy = cql_q2_policy - cql_policy_log_probs

            # We need to broadcast the scalar log_prob to the tensor shape [N*k, T, 1]
            cql_q1_uniform = cql_q1_uniform - cql_uniform_log_probs
            cql_q2_uniform = cql_q2_uniform - cql_uniform_log_probs

            # Concatenate policy and uniform samples along the sample dimension
            cql_q1_cat = torch.cat([reshape_cql(cql_q1_policy), reshape_cql(cql_q1_uniform)], dim=-1)
            cql_q2_cat = torch.cat([reshape_cql(cql_q2_policy), reshape_cql(cql_q2_uniform)], dim=-1)

            # Calculate logsumexp: shape [N, T, 1]
            logsumexp_q1 = torch.logsumexp(cql_q1_cat, dim=-1, keepdim=True)
            logsumexp_q2 = torch.logsumexp(cql_q2_cat, dim=-1, keepdim=True)

            # --- 5. Combine Losses and Apply Visible/Next_Visible Mask ---

            if self.decoy_interval == 0:
                # --- N-Step Return Mode: Filter all loss components to [M, 1] ---

                # Get selected Bellman terms (shape [M, 1]) and the mask (shape [N, T])
                q1_pred_sel, q2_pred_sel, q_target_sel, valid_mask = self._filter_to_correct_visibles(
                    q1_pred, q2_pred, r, next_v, dones, visible, next_visible, padding_mask, train_mask)

                # Calculate Bellman loss on selected elements
                bellman_loss = F.mse_loss(q1_pred_sel, q_target_sel, reduction='none') + \
                               F.mse_loss(q2_pred_sel, q_target_sel, reduction='none')  # Shape [M, 1]

                # --- Filter CQL components to match ---
                # Use the mask returned by the filter function
                valid_mask_expanded = valid_mask.unsqueeze(-1)  # Shape [N, T, 1]

                # Select the logsumexp terms for the *same* M timesteps
                logsumexp_q1_sel = logsumexp_q1[valid_mask_expanded].view(-1, 1)  # Shape [M, 1]
                logsumexp_q2_sel = logsumexp_q2[valid_mask_expanded].view(-1, 1)  # Shape [M, 1]

                # Calculate CQL loss on selected elements
                cql_loss1 = logsumexp_q1_sel - q1_pred_sel
                cql_loss2 = logsumexp_q2_sel - q2_pred_sel
                cql_loss = cql_loss1 + cql_loss2  # Shape [M, 1]

                # Combine losses
                total_loss_unmasked = bellman_loss + self._cql_alpha * cql_loss  # Shape [M, 1]

                # The loss is already filtered, just take the mean
                total_loss = total_loss_unmasked.mean()

            else:
                # --- 1-Step Return Mode: Calculate on [N, T, 1] and mask at the end ---

                # Calculate 1-step target for *all* timesteps
                q_target = r + self._gamma * (1 - dones.float()) * next_v  # Shape [N, T, 1]

                # Calculate Bellman loss for *all* timesteps
                bellman_loss = F.mse_loss(q1_pred, q_target, reduction='none') + \
                               F.mse_loss(q2_pred, q_target, reduction='none')  # Shape [N, T, 1]

                # Calculate CQL Loss for *all* timesteps
                cql_loss1 = logsumexp_q1 - q1_pred
                cql_loss2 = logsumexp_q2 - q2_pred
                cql_loss = cql_loss1 + cql_loss2  # Shape [N, T, 1]

                # Combine losses
                total_loss_unmasked = bellman_loss + self._cql_alpha * cql_loss  # Shape [N, T, 1]

                # --- Apply Masking ---
                # Use train_mask (which is False for padding AND burn-in)
                final_mask = train_mask.squeeze(-1)  # Shape (N, T)
                masked_loss = (total_loss_unmasked.squeeze(-1) * final_mask.float())
                total_loss = masked_loss.sum() / (final_mask.sum() + self._eps)

        self.critic_optim.zero_grad()
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.critic_optim)
            self.scaler.update()
        else:
            total_loss.backward()
            self.critic_optim.step()
        return total_loss

    def _update_actor(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask, train_mask):
        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            (action_mean, log_std), _ = self.policy_net(obs, padding_mask=padding_mask, train_mask=train_mask)

            # Convert to Normal distribution
            std = log_std.exp()
            base_dist = Normal(action_mean, std)

            # Sample x (pre-tanh) using the reparameterization trick
            x_sampled = base_dist.rsample()

            # Apply the tanh(a*x) squash
            u_sampled = torch.tanh(x_sampled * self._tanh_scale)

            # If using decoy_interval == 0, persist actions over non-visible steps
            if self.decoy_interval == 0:
                u_sampled_persistent = _persist_actions(u_sampled, visible)
            else:
                u_sampled_persistent = u_sampled

            # Calculate log_prob of the action u_sampled
            log_jacob = torch.log(1 - u_sampled.pow(2) + self._eps) + self.log_a
            log_pi_sampled = base_dist.log_prob(x_sampled) - log_jacob
            log_pi_sampled = log_pi_sampled.sum(dim=-1, keepdim=True)

            # Get Q-values for these new actions
            # We pass u_sampled (normalized action) to the critics, as they expect.
            # We do *not* detach gradients here, as we need them to flow through Q into u_sampled
            q1_values, _ = self.critic_net1(obs, u_sampled_persistent, padding_mask=padding_mask, train_mask=train_mask)
            q2_values, _ = self.critic_net2(obs, u_sampled_persistent, padding_mask=padding_mask, train_mask=train_mask)
            q_values = torch.min(q1_values, q2_values)

            # Calculate SAC Actor Loss
            # The loss is (alpha * log_prob) - Q_value
            policy_loss_unmasked = (self._entropy_alpha * log_pi_sampled - q_values)

            final_mask = train_mask.squeeze(-1)
            if self.decoy_interval == 0:
                final_mask = final_mask & visible.squeeze(-1)

            masked_loss = (policy_loss_unmasked.squeeze(-1) * final_mask.float())
            policy_loss = masked_loss.sum() / (final_mask.sum() + self._eps)

        self.policy_optim.zero_grad()
        if self.scaler:
            self.scaler.scale(policy_loss).backward()
            self.scaler.step(self.policy_optim)
            self.scaler.update()
        else:
            policy_loss.backward()
            self.policy_optim.step()

        return policy_loss

    def sync_target_networks(self, tau=None):
        tau = tau or self._tau_target
        for target_param, param in zip(self.target_critic_net1.parameters(), self.critic_net1.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
        for target_param, param in zip(self.target_critic_net2.parameters(), self.critic_net2.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)


@torch.compile
def _persist_actions(actions, visible_mask):
    # We must clone to avoid modifying the original tensor
    persisted_actions = actions.clone()

    B, T, A_dim = persisted_actions.shape
    device = persisted_actions.device

    # 1. Create a tensor of indices [0, 1, 2, ..., T-1]
    # Shape: [1, T, 1]
    indices = torch.arange(T, device=device).view(1, T, 1)

    # (Where not visible, set index to 0)
    masked_indices = torch.where(
        visible_mask,
        indices,
        0  # Set to 0 where not visible
    )

    # 3. Apply cummax to forward-fill the last valid index
    # Shape: [B, T, 1]
    filled_indices = torch.cummax(masked_indices, dim=1).values

    # 4. Expand indices to match action dim and gather
    # Shape: [B, T, A_dim]
    filled_indices_expanded = filled_indices.expand(-1, -1, A_dim)

    # Gather values from the actions
    # This single operation replaces the entire loop
    persisted_actions = torch.gather(persisted_actions, 1, filled_indices_expanded)

    return persisted_actions
