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
from torch.distributions import Normal, Beta, TransformedDistribution, TanhTransform, AffineTransform
from stable_baselines3.common.distributions import sum_independent_dims
from tqdm import tqdm
from gym_wrappers import TOTAL_SIZE
from ppo_trainer import INSULIN_ACTION_LOW, INSULIN_ACTION_HIGH


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


class CustomBetaDistribution(nn.Module):
    """
    Beta distribution, scaled to an arbitrary range [low, high].

    This distribution is naturally bounded, making it a good fit for
    action spaces with hard constraints.

    :param action_dim: Dimension of the action space.
    :param low: The lower bound of the action space.
    :param high: The upper bound of the action space.
    """

    def __init__(self, action_dim: int, low: float = 0.0, high: float = 2.0):
        super().__init__()
        self.action_dim = action_dim
        self.scale = float(high - low)
        self.bias = float(low)
        self.epsilon = 1e-6
        self.distribution = None
        self.alpha = None
        self.beta = None

        # Pre-compute log_scale for log_prob calculation
        self.log_scale = torch.tensor(self.scale).log()

    def proba_distribution(self, params: torch.Tensor, _=None):
        """
        Creates the distribution from the actor network's output.

        :param params: Raw output from the actor head (batch_size, action_dim * 2)
        """
        # Split the output into alpha and beta parameters
        # We add 1.0 + softplus(x) to ensure alpha and beta are > 1,
        # which makes the distribution more stable and uni-modal to start.
        # F.softplus(x) = log(1 + exp(x))
        alpha, beta = torch.chunk(params, 2, dim=-1)
        self.alpha = F.softplus(alpha) + 1.0
        self.beta = F.softplus(beta) + 1.0

        # Create the underlying Beta distribution
        self.distribution = Beta(self.alpha, self.beta)
        return self

    def entropy(self) -> torch.Tensor:
        """
        Get the entropy of the distribution.
        Entropy of a scaled distribution H(aX + b) = H(X) + log|a|
        """
        # Get entropy from the base Beta distribution
        entropy = self.distribution.entropy()

        # Sum entropy across action dimensions
        entropy = sum_independent_dims(entropy)

        # Add the scaling factor's log-determinant
        # (we add log_scale for each dimension)
        entropy += self.log_scale * self.action_dim
        return entropy

    def sample(self) -> torch.Tensor:
        """
        Sample an action, using the reparameterization trick (rsample).
        """
        # Sample from Beta(alpha, beta) -> range [0, 1]
        u = self.distribution.rsample()

        # Scale and shift to [low, high]
        return self.scale * u + self.bias

    def mode(self) -> torch.Tensor:
        """
        Return the mode of the distribution (most likely action).
        For Beta(a,b) with a,b > 1, mode is (a-1)/(a+b-2).
        We use the mean as a stable approximation.
        Mean is a / (a + b)
        """
        u = self.distribution.mean  # range [0, 1]

        # Scale and shift to [low, high]
        return self.scale * u + self.bias

    def log_prob(self, actions: torch.Tensor, gaussian_actions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Get the log-probability of taking a specific action.

        :param actions: The actions taken (in range [low, high])
        :param gaussian_actions: Ignored (kept for API compatibility)
        """
        # Un-scale the actions from [low, high] back to [0, 1]
        u = (actions - self.bias) / self.scale

        # Clamp actions to be slightly inside (0, 1) for numerical stability
        u = torch.clamp(u, self.epsilon, 1.0 - self.epsilon)

        # Get log-prob from the base Beta distribution
        log_prob = self.distribution.log_prob(u)
        log_prob = sum_independent_dims(log_prob)

        # Account for the scaling transformation
        # log_prob(action) = log_prob(u) - log(scale)
        log_prob -= self.log_scale * self.action_dim
        return log_prob


class SharedRecurrentEncoder(nn.Module):
    """
    This module encapsulates the shared parts of the network:
    1. The observation encoder (e.g., MiniGridEncoder)
    2. The LSTM layer

    It takes raw observations and outputs the
    sequence of LSTM hidden states (history-aware state features).
    """

    def __init__(self, observation_shape: Tuple[int, int, int], hidden_dim: int = 128,
                 recurrent_hidden_size: int = 128, device: str = 'cpu',
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

        self.burn_in_length = burn_in_length

        # --- Recurrent Layer ---
        # The LSTM input is now *only* the feature dimension from the encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=recurrent_hidden_size,
            num_layers=1,
            batch_first=True  # Crucial for easier tensor manipulation
        ).to(device)

    @torch.compile
    def forward(self, x: Optional[torch.Tensor] = None,
                obs_features: Optional[torch.Tensor] = None,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                padding_mask: Optional[torch.Tensor] = None,
                train_mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processes observations through the encoder and LSTM.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - lstm_out: The sequence of LSTM hidden states, shape (N, T, recurrent_hidden_size)
            - next_hidden_state: The final hidden state (h_n, c_n) for the *next* step.
        """

        # Get our features shape
        if obs_features is not None:
            # features shape: (N, T, feature_dim)
            batch_size, seq_len = obs_features.shape[0], obs_features.shape[1]
            # (N, T, feature_dim) -> (N * T, feature_dim)
            hidden = obs_features.reshape(-1, obs_features.shape[2])
        elif x is not None:
            # x shape: (N, T, C, H, W)
            batch_size, seq_len = x.shape[0], x.shape[1]
            # (N, T, C, H, W) -> (N * T, C, H, W)
            x_reshaped = x.reshape(-1, *x.shape[2:])
            # (N * T, C, H, W) -> (N * T, hidden_dim)
            hidden = self.encoder(x_reshaped.to(dtype=torch.float32))
        else:
            raise ValueError("Either 'x' (observations) or 'obs_features' (pre-computed) must be provided.")

        # --- Action logic has been removed ---
        # Actions will be fused by the critic *after* this module

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

        return lstm_out, next_hidden_state

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


class RecurrentNet(nn.Module):
    """
    This module is now just a decoder.
    It takes a sequence of features and passes them through
    a final MLP to produce the desired output.

    For Value/Policy nets, input_features = lstm_state_features
    For Critic nets, input_features = torch.cat([lstm_state_features, action_features], dim=-1)
    """

    def __init__(self, input_feature_size: int, output_size: int, has_action_encoder: bool = False,
                 device: str = 'cpu', dropout_p: float = 0.0, *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        self.input_feature_size = input_feature_size
        self._has_action_encoder = has_action_encoder

        if self._has_action_encoder:
            self.action_encoder = nn.Sequential(
                nn.Linear(1, input_feature_size // 2),
                nn.LayerNorm(input_feature_size // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(input_feature_size // 2, input_feature_size)
            )
        else:
            self.action_encoder = None

        self.decoder = nn.Sequential(
            nn.Linear(input_feature_size, input_feature_size // 2),
            nn.LayerNorm(input_feature_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(input_feature_size // 2, output_size)
        ).to(device)

        # Note: Weight initialization is omitted for brevity but should be the same as in your original code.
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

    @torch.compile
    def forward(self, input_features: torch.Tensor,
                actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processes a sequence of input features through the decoder.

        Args:
            input_features (torch.Tensor): Input features,
                                           shape (N, T, input_feature_size)
            actions (torch.Tensor, optional): Actions to be concatenated,
                                              shape (N, T, action_feature_size)

        Returns:
            torch.Tensor: The final decoded output, shape (N, T, output_size)
        """
        assert actions is None or self._has_action_encoder, \
            "Action encoder must be provided if actions are used."

        if actions is not None:
            encoded_actions = self.action_encoder(actions)
            input_features = input_features + encoded_actions

        batch_size, seq_len = input_features.shape[0], input_features.shape[1]

        # Reshape for the decoder
        # (N, T, input_feature_size) -> (N * T, input_feature_size)
        decoder_input = input_features.reshape(-1, self.input_feature_size)

        # (N * T, input_feature_size) -> (N * T, output_size)
        output = self.decoder(decoder_input)

        # Reshape final output back to sequence format
        # (N * T, output_size) -> (N, T, output_size)
        output = output.view(batch_size, seq_len, -1)

        return output


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
        self._hidden_dim = hidden_dim
        self._recurrent_hidden_size = self._hidden_dim if recurrent_hidden_size is None else recurrent_hidden_size
        self._batch_size = batch_size
        self.dist = CustomBetaDistribution(action_dim=1, low=INSULIN_ACTION_LOW, high=INSULIN_ACTION_HIGH)
        self._eps = 1e-5
        self._tanh_scale = 1.0

        # --- PLACEHOLDERS: These must be set by the child class ---
        self.shared_encoder: SharedRecurrentEncoder = None
        self.policy_net: RecurrentNet = None
        # ---

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
        # Add batch dimension
        obs_tensor = obs_tensor.unsqueeze(0)

        # Pass the current hidden_state to the policy network
        # 1. Call the shared encoder
        #    Input: (1, 1, H, W, C), Output: (1, 1, hidden_size)
        lstm_out, next_hidden_state = self.shared_encoder(obs_tensor,
                                                          hidden_state=hidden_state,
                                                          padding_mask=None,
                                                          train_mask=None)

        # 2. Call the policy decoder
        #    Input: (1, 1, hidden_size), Output: (1, 1, 2)
        params = self.policy_net(lstm_out, actions=None)

        dist = self.dist.proba_distribution(params)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action.squeeze(-1).cpu().numpy(), next_hidden_state

    @torch.compile
    def _filter_to_correct_visibles(self, q1, q2, r, v_next, dones, visible, next_visible, padding_mask, train_mask):
        # shapes: q1,q2 (N,T,1 or N,T,*), r,v_next,dones,visible,next_visible (N,T,1)
        N, T = visible.shape[:2]
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

    def get_initial_states(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a zero-initialized hidden state tuple for the LSTM."""
        h_0 = torch.zeros(1, batch_size, self._recurrent_hidden_size, device=self._device)
        c_0 = torch.zeros(1, batch_size, self._recurrent_hidden_size, device=self._device)
        return h_0, c_0

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

        # Kwargs for encoders and decoders
        encoder_kwargs = dict(
            observation_shape=self._observation_shape,
            hidden_dim=self._hidden_dim,
            recurrent_hidden_size=self._recurrent_hidden_size,
            dropout_p=dropout_p,
            device=self._device
        )

        decoder_kwargs = dict(
            input_feature_size=self._recurrent_hidden_size,
            dropout_p=dropout_p,
            device=self._device
        )

        # --- Create Shared Encoders ---
        self.feature_extractor = OfflineMiniGridEncoder(**encoder_kwargs).to(self._device)
        self.target_feature_extractor = OfflineMiniGridEncoder(**encoder_kwargs).to(self._device)

        self.shared_encoder = SharedRecurrentEncoder(
            feature_extractor=self.feature_extractor,
            **encoder_kwargs
        ).to(self._device)
        self.target_shared_encoder = SharedRecurrentEncoder(
            feature_extractor=self.target_feature_extractor,
            **encoder_kwargs
        ).to(self._device)

        # --- Create Decoders ---
        self.critic_net1 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)
        self.critic_net2 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)
        self.value_net = RecurrentNet(output_size=1, **decoder_kwargs).to(self._device)
        self.target_value_net = RecurrentNet(output_size=1, **decoder_kwargs).to(self._device)
        self.policy_net = RecurrentNet(output_size=2, **decoder_kwargs).to(self._device)

        # --- Setup Optimizers ---
        # Shared parameters will be optimized by all three optimizers
        shared_params = list(self.shared_encoder.parameters())

        self.critic_optim = torch.optim.AdamW(
            shared_params + list(self.critic_net1.parameters()) + list(self.critic_net2.parameters()),
            lr=self._critic_lr
        )
        self.value_optim = torch.optim.AdamW(
            shared_params + list(self.value_net.parameters()),
            lr=self._value_lr
        )
        self.policy_optim = torch.optim.AdamW(
            shared_params + list(self.policy_net.parameters()),
            lr=self._actor_lr
        )

        # clone the value net to a target network
        self.sync_target_networks(tau=1.0)

    def forward(self, obs, acts):
        # Note that this returns both the output and the hidden states
        lstm_out, _ = self.shared_encoder(obs)
        q1 = self.critic_net1(lstm_out, actions=acts)
        q2 = self.critic_net2(lstm_out, actions=acts)

        v = self.value_net(lstm_out)
        logits = self.policy_net(lstm_out)

        return (q1, q2), v, logits

    def sync_target_networks(self, tau=None):
        tau = tau or self._tau_target
        # Sync the target value decoder
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

        # Sync the target shared encoder
        for target_param, param in zip(self.target_shared_encoder.parameters(), self.shared_encoder.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    @torch.compile
    def _update_critic(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                       train_mask):
        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            # We only need the network output, not the final hidden state for training
            lstm_out, hidden_state = self.shared_encoder(obs, padding_mask=padding_mask, train_mask=train_mask)
            q1 = self.critic_net1(lstm_out, actions=acts)
            q2 = self.critic_net2(lstm_out, actions=acts)

            with torch.no_grad():
                # Use target shared encoder and target value decoder
                next_lstm_out, _ = self.target_shared_encoder(next_obs, padding_mask=next_padding_mask, train_mask=None)
                v_next = self.target_value_net(next_lstm_out, actions=None)
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
    def _update_value(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                      train_mask):
        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            # Value function doesn't depend on actions
            lstm_out, _ = self.shared_encoder(obs, padding_mask=padding_mask, train_mask=train_mask)
            v = self.value_net(lstm_out, actions=None)

            with torch.no_grad():
                # Q-values *do* depend on actions
                # lstm_out_q, _ = self.shared_encoder(obs, actions=acts, padding_mask=padding_mask)
                q1 = self.critic_net1(lstm_out, actions=acts)
                q2 = self.critic_net2(lstm_out, actions=acts)
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
    def _update_actor(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                      train_mask):
        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            # Policy doesn't depend on actions
            lstm_out, _ = self.shared_encoder(obs, padding_mask=padding_mask, train_mask=train_mask)
            params = self.policy_net(lstm_out, actions=None)

            dist = self.dist.proba_distribution(params)

            # Calculate the log-probability of the expert actions `acts`
            #    under the new Beta distribution.
            #    The `dist.log_prob` method handles all transformations and scaling.
            log_pi = dist.log_prob(acts)

            # 2. The policy loss is the Negative Log-Likelihood (NLL)
            policy_loss_unmasked = -log_pi

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
            tau_target: float = 0.005,
            entropy_alpha: float = 0.2,
            dropout_p: float = 0.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._tau_target = tau_target
        self._cql_target = 5.0
        # We learn the log of alpha for numerical stability
        self.log_cql_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self._entropy_alpha = entropy_alpha
        self.cql_uniform_log_probs = -torch.log(torch.tensor(INSULIN_ACTION_HIGH - INSULIN_ACTION_LOW,
                                                             device=self._device))

        # Kwargs for encoders and decoders
        encoder_kwargs = dict(
            observation_shape=self._observation_shape,
            hidden_dim=self._hidden_dim,
            recurrent_hidden_size=self._recurrent_hidden_size,
            dropout_p=dropout_p,
            device=self._device
        )

        decoder_kwargs = dict(
            input_feature_size=self._recurrent_hidden_size,
            dropout_p=dropout_p,
            device=self._device
        )

        self.feature_extractor = OfflineMiniGridEncoder(**encoder_kwargs).to(self._device)
        self.target_feature_extractor = OfflineMiniGridEncoder(**encoder_kwargs).to(self._device)

        # --- Create Shared Encoders ---
        self.shared_encoder = SharedRecurrentEncoder(
            feature_extractor=self.feature_extractor,
            **encoder_kwargs
        ).to(self._device)
        self.target_shared_encoder = SharedRecurrentEncoder(
            feature_extractor=self.target_feature_extractor,
            **encoder_kwargs
        ).to(self._device)

        # --- Create Decoders ---
        self.critic_net1 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)
        self.critic_net2 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)
        self.target_critic_net1 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)
        self.target_critic_net2 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)
        self.policy_net = RecurrentNet(output_size=2, **decoder_kwargs).to(self._device)
        self.cql_alpha_optim = torch.optim.AdamW([self.log_cql_alpha], lr=self._critic_lr)

        # --- Setup Optimizers ---
        shared_params = list(self.shared_encoder.parameters())

        self.critic_optim = torch.optim.AdamW(
            shared_params + list(self.critic_net1.parameters()) + list(self.critic_net2.parameters()),
            lr=self._critic_lr
        )
        self.policy_optim = torch.optim.AdamW(
            shared_params + list(self.policy_net.parameters()),
            lr=self._actor_lr
        )

        # Initialize target networks
        self.sync_target_networks(tau=1.0)

    def forward(self, obs, acts):
        # Note that this returns both the output and the hidden states
        lstm_out, _ = self.shared_encoder(obs)

        q1_logits = self.critic_net1(lstm_out, actions=acts)
        q2_logits = self.critic_net2(lstm_out, actions=acts)
        policy_logits = self.policy_net(lstm_out)

        return q1_logits, q2_logits, policy_logits

    @torch.compile
    def _update_critic(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                       train_mask):
        def reshape_cql(tensor):
            return tensor.view(batch_size, cql_n_samples, seq_len, 1).permute(0, 2, 1, 3).squeeze(-1)

        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            # --- Get Q-values for actions in the dataset ---
            lstm_out, _ = self.shared_encoder(obs, padding_mask=padding_mask, train_mask=train_mask)
            q1_pred = self.critic_net1(lstm_out, actions=acts)
            q2_pred = self.critic_net2(lstm_out, actions=acts)

            with torch.no_grad():
                # We detach lstm_out to prevent critic gradients from flowing through here to the policy
                params_data = self.policy_net(lstm_out.detach(), actions=None)
                dist_data = self.dist.proba_distribution(params_data)
                # Get log_prob of the *dataset action* (acts) under the current policy
                log_pi_data = dist_data.log_prob(acts).unsqueeze(-1)  # Shape [N, T, 1]

            # --- Calculate Bellman Target Components ---
            with torch.no_grad():
                # Get policy params from *online* policy net and *online* encoder
                next_lstm_out, _ = self.shared_encoder(next_obs, padding_mask=next_padding_mask, train_mask=None)
                params = self.policy_net(next_lstm_out, actions=None)

                # Create the Beta distribution
                dist = self.dist.proba_distribution(params)

                # Sample next action using reparameterization trick
                # This action is already in the [low, high] range
                next_actions = dist.sample()

                # If using decoy_interval == 0, persist actions over non-visible steps
                if self.decoy_interval == 0:
                    next_actions_persistent = _persist_actions(next_actions, next_visible)
                else:
                    next_actions_persistent = next_actions

                # Calculate log_prob
                # The dist.log_prob method handles all scaling and transformations.
                # We unsqueeze to get shape [N, T, 1] as log_prob returns [N, T]
                next_log_probs = dist.log_prob(next_actions).unsqueeze(-1)

                # Get Q-values from *target* critic decoders and *target* shared encoder
                next_lstm_out_target, _ = self.target_shared_encoder(next_obs, padding_mask=next_padding_mask, train_mask=None)
                next_q1 = self.target_critic_net1(next_lstm_out_target, actions=next_actions_persistent)
                next_q2 = self.target_critic_net2(next_lstm_out_target, actions=next_actions_persistent)
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

            # Get policy params from *online* policy net and *online* encoder
            cql_lstm_out_policy, _ = self.shared_encoder(cql_obs,
                                                         padding_mask=cql_padding_mask,
                                                         train_mask=cql_train_mask)

            with torch.no_grad():
                params = self.policy_net(cql_lstm_out_policy, actions=None)

                # Create the Beta distribution
                dist = self.dist.proba_distribution(params)

                # Sample actions (already in [low, high] range)
                cql_policy_actions = dist.sample()

                # Calculate log prob
                cql_policy_log_probs = dist.log_prob(cql_policy_actions).unsqueeze(-1)

            # Get the [low, high] bounds from the distribution instance
            low = self.dist.bias
            high = self.dist.bias + self.dist.scale

            cql_uniform_actions = torch.empty(batch_size * cql_n_samples, seq_len, 1, device=obs.device).uniform_(low, high)

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

            # --- Get Q-values for all sampled actions (using online encoder/decoders) ---
            cql_q1_policy = self.critic_net1(cql_lstm_out_policy, actions=cql_policy_actions_persistent)
            cql_q2_policy = self.critic_net2(cql_lstm_out_policy, actions=cql_policy_actions_persistent)

            cql_q1_uniform = self.critic_net1(cql_lstm_out_policy, actions=cql_uniform_actions_persistent)
            cql_q2_uniform = self.critic_net2(cql_lstm_out_policy, actions=cql_uniform_actions_persistent)

            # --- Apply log-prob correction (log Q - log pi) ---
            cql_q1_policy = cql_q1_policy - cql_policy_log_probs
            cql_q2_policy = cql_q2_policy - cql_policy_log_probs

            # We need to broadcast the scalar log_prob to the tensor shape [N*k, T, 1]
            cql_q1_uniform = cql_q1_uniform - self.cql_uniform_log_probs
            cql_q2_uniform = cql_q2_uniform - self.cql_uniform_log_probs

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

                # Filter log_pi_data to match
                log_pi_data_sel = log_pi_data[valid_mask_expanded].view(-1, 1)

                # Select the logsumexp terms for the *same* M timesteps
                logsumexp_q1_sel = logsumexp_q1[valid_mask_expanded].view(-1, 1)  # Shape [M, 1]
                logsumexp_q2_sel = logsumexp_q2[valid_mask_expanded].view(-1, 1)  # Shape [M, 1]

                # Calculate CQL loss on selected elements
                cql_term1_data = q1_pred_sel - log_pi_data_sel
                cql_term2_data = q2_pred_sel - log_pi_data_sel

                cql_diff1 = logsumexp_q1_sel - cql_term1_data - self._cql_target
                cql_diff2 = logsumexp_q2_sel - cql_term2_data - self._cql_target

                cql_loss1 = F.relu(cql_diff1)
                cql_loss2 = F.relu(cql_diff2)
                cql_loss = cql_loss1 + cql_loss2  # Shape [M, 1]

                # Combine losses
                cql_alpha = self.log_cql_alpha.exp().detach()
                total_loss_unmasked = bellman_loss + cql_alpha * cql_loss  # Shape [M, 1]

                # --- NEW: Calculate and update log_cql_alpha ---
                # The loss for alpha is the negative of the *mean* raw difference,
                # weighted by the (non-detached) alpha.
                # We want to increase alpha if (cql_diff - target) > 0
                cql_diff_mean = (cql_diff1.mean() + cql_diff2.mean()) * 0.5
                alpha_loss = -torch.exp(self.log_cql_alpha) * (cql_diff_mean - self._cql_target).detach()

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
                cql_term1_data = q1_pred - log_pi_data  # Shape [N, T, 1]
                cql_term2_data = q2_pred - log_pi_data  # Shape [N, T, 1]

                cql_diff1 = logsumexp_q1 - cql_term1_data - self._cql_target
                cql_diff2 = logsumexp_q2 - cql_term2_data - self._cql_target

                cql_loss1 = F.relu(cql_diff1)
                cql_loss2 = F.relu(cql_diff2)
                cql_loss = cql_loss1 + cql_loss2  # Shape [N, T, 1]

                # Combine losses
                cql_alpha = self.log_cql_alpha.exp().detach()
                total_loss_unmasked = bellman_loss + cql_alpha * cql_loss  # Shape [N, T, 1]

                # --- Apply Masking ---
                # Use train_mask (which is False for padding AND burn-in)
                final_mask = train_mask.squeeze(-1).float()  # Shape (N, T)
                valid_count = final_mask.sum() + self._eps
                masked_loss = (total_loss_unmasked.squeeze(-1) * final_mask)
                total_loss = masked_loss.sum() / valid_count

                # do the same for the alpha loss
                masked_cql_diff1 = (cql_diff1.squeeze(-1) * final_mask)
                masked_cql_diff2 = (cql_diff2.squeeze(-1) * final_mask)
                # Calculate the masked mean of the diffs
                cql_diff1_mean = masked_cql_diff1.sum() / valid_count
                cql_diff2_mean = masked_cql_diff2.sum() / valid_count
                cql_diff_mean = (cql_diff1_mean + cql_diff2_mean) * 0.5
                alpha_loss = -torch.exp(self.log_cql_alpha) * cql_diff_mean.detach()

        # Update critic and alpha
        self.critic_optim.zero_grad()
        self.cql_alpha_optim.zero_grad()
        if self.scaler:
            self.scaler.scale(total_loss + alpha_loss).backward()
            self.scaler.unscale_(self.critic_optim)
            self.scaler.unscale_(self.cql_alpha_optim)
            torch.nn.utils.clip_grad_norm_(self.critic_net1.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic_net2.parameters(), 0.5)
            self.scaler.step(self.critic_optim)
            self.scaler.step(self.cql_alpha_optim)
            self.scaler.update()
        else:
            total_loss.backward()
            self.critic_optim.step()
        return total_loss

    @torch.compile
    def _update_actor(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask, train_mask):
        with torch.autocast(device_type="cuda") if self.scaler is not None else nullcontext():
            # 1. Get policy params (action-independent)
            lstm_out, _ = self.shared_encoder(obs, padding_mask=padding_mask, train_mask=train_mask)
            params = self.policy_net(lstm_out, actions=None)

            # Create the Beta distribution
            dist = self.dist.proba_distribution(params)

            # Sample an action (rsample) in [low, high] range
            # This is the new 'actions_sampled'
            actions_sampled = dist.sample()

            # If using decoy_interval == 0, persist actions over non-visible steps
            if self.decoy_interval == 0:
                actions_sampled_persistent = _persist_actions(actions_sampled, visible)
            else:
                actions_sampled_persistent = actions_sampled

            # Calculate log_prob of the sampled action
            # dist.log_prob() returns [N, T], so unsqueeze to [N, T, 1]
            log_pi_sampled = dist.log_prob(actions_sampled).unsqueeze(-1)

            # Get Q-values for these new actions
            # We pass u_sampled (normalized action) to the critics, as they expect.
            # We do *not* detach gradients here, as we need them to flow through Q into u_sampled
            q1_values = self.critic_net1(lstm_out, actions=actions_sampled_persistent)
            q2_values = self.critic_net2(lstm_out, actions=actions_sampled_persistent)
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
            self.scaler.unscale_(self.policy_optim)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.scaler.step(self.policy_optim)
            self.scaler.update()
        else:
            policy_loss.backward()
            self.policy_optim.step()

        return policy_loss

    def sync_target_networks(self, tau=None):
        tau = tau or self._tau_target
        # Sync target critic decoders
        for target_param, param in zip(self.target_critic_net1.parameters(), self.critic_net1.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
        for target_param, param in zip(self.target_critic_net2.parameters(), self.critic_net2.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

        # Sync target shared encoder
        for target_param, param in zip(self.target_shared_encoder.parameters(), self.shared_encoder.parameters()):
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
