from collections import deque
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from tqdm import tqdm
from scipy.stats import trimboth

from ppo_trainer import INSULIN_ACTION_LOW, INSULIN_ACTION_HIGH, set_seed

torch._dynamo.config.capture_dynamic_output_shape_ops = True


class FeatureEncoder(nn.Module):
    """
    A feature extractor that uses self-attention on the input features.

    This network treats the input features as a sequence and uses a
    Multi-Head Attention block to learn the relationships between them.

    :param input_dim: The input dimension.
    :param hidden_dim: The final output dimension of the extractor.
    """

    def __init__(
            self,
            input_dim,
            hidden_dim: int = 128,
            *args,
            **kwargs
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
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
        # Assuming x has shape (batch_size, num_inputs)
        return self.fc(x)


class CustomBetaDistribution(nn.Module):
    """
    Beta distribution, scaled to an arbitrary range [low, high].

    This distribution is naturally bounded, making it a good fit for
    action spaces with hard constraints.

    :param action_dim: Dimension of the action space.
    :param low: The lower bound of the action space.
    :param high: The upper bound of the action space.
    """

    def __init__(self, action_dim: int = 1, low: float = 0.0, high: float = 2.0):
        super().__init__()
        self.action_dim = action_dim
        self.low = low
        self.high = high

        self.scale = float(high - low)
        self.bias = float(low)

        self.epsilon = 1e-6
        self.distribution = None
        self.alpha = None
        self.beta = None
        self.mu = None
        self.kappa = None

        # --- Constants for numerical stability ---
        self.clamp_max = 100.0

        # Pre-compute log_scale for log_prob calculation
        self.register_buffer("log_scale", torch.tensor(self.scale).log())

    def proba_distribution(self, params: torch.Tensor):
        """
        Creates the distribution from the actor network's output.

        :param params: Raw output from the actor head (batch_size, action_dim * 2)
        """
        # Split the output into alpha and beta parameters
        mu_logits, kappa_logits = torch.chunk(params, 2, dim=-1)

        self.mu = torch.sigmoid(mu_logits)
        self.kappa = torch.exp(kappa_logits) + 1.0

        # Softplus to keep alpha, beta > 1 for unimodal distribution
        self.alpha = self.mu * self.kappa
        self.beta = (1 - self.mu) * self.kappa

        # (optional) Clamp for numerical stability
        # self.alpha = torch.clamp(self.alpha, 1.0, self.clamp_max)
        # self.beta = torch.clamp(self.beta, 1.0, self.clamp_max)

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

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log-probability of taking a specific action.

        :param actions: The actions taken (in range [low, high])
        """
        # Un-scale the actions from [low, high] back to [0, 1]
        u = (actions - self.bias) / self.scale

        # Clamp actions to be slightly inside (0, 1) for numerical stability
        u = torch.clamp(u, self.epsilon, 1.0 - self.epsilon)

        # Get log-prob from the base Beta distribution
        log_prob = self.distribution.log_prob(u)

        # Account for the scaling transformation
        log_prob -= self.log_scale * self.action_dim
        return log_prob.sum(dim=-1, keepdim=True)

    def sample_and_logprob(self):
        """
        Samples an action using the reparameterization trick AND computes
        its log-probability in a single, numerically stable pass.

        :return: (actions, log_prob)
        """
        # 1. Sample from Beta(alpha, beta) -> range [0, 1]
        u = self.distribution.rsample()

        # 2. Scale and shift to [low, high]
        actions = self.scale * u + self.bias

        # --- Calculate log_prob from u ---

        # 1. Clamp u for numerical stability before log_prob
        u_clamped = torch.clamp(u, self.epsilon, 1.0 - self.epsilon)

        # 2. Get log-prob from the base Beta distribution
        log_prob_u = self.distribution.log_prob(u_clamped)

        # 3. Apply the change of variables formula
        # log_prob(action) = log_prob(u) - log(scale)
        log_pi = log_prob_u - self.log_scale * self.action_dim

        return actions, log_pi.sum(dim=-1, keepdim=True)

    def variance(self):
        mu = self.mu.detach()
        variance = (mu * (1-mu)) / (self.kappa + 1)
        return variance


class SquashedGaussianDistribution(nn.Module):
    """
    Squashed Gaussian distribution.

    A Gaussian distribution is sampled, passed through a tanh squashing function
    to bound the output to [-1, 1], and then scaled and shifted to fit the
    arbitrary range [low, high].

    This is commonly used in policy gradient methods (e.g., SAC) for
    continuous action spaces with hard constraints.

    :param action_dim: Dimension of the action space.
    :param low: The lower bound of the action space.
    :param high: The upper bound of the action space.
    """

    def __init__(self, action_dim: int, low: float = 0.0, high: float = 2.0):
        super().__init__()
        self.action_dim = action_dim
        self.low = low
        self.high = high

        self.scale = (high - low) / 2.0
        self.bias = (high + low) / 2.0

        self.epsilon = 1e-6
        self.distribution = None
        self.mu = None
        self.std = None

        # --- Constants for numerical stability ---
        self.log_std_min = -10.0
        self.log_std_max = 2.0

        # Pre-compute log_scale and register as a buffer
        # This is log( (high - low) / 2 )
        self.register_buffer("log_scale", torch.tensor(self.scale).log())

    def proba_distribution(self, params: torch.Tensor):
        """
        Creates the distribution from the actor network's output.

        :param params: Raw output from the actor head (batch_size, action_dim * 2)
                       Assumed to be [mu, log_std].
        """
        # Split the output into mu and log_std
        self.mu, log_std = torch.chunk(params, 2, dim=-1)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # Calculate std
        self.std = log_std.exp()

        # Create the underlying Normal (Gaussian) distribution
        self.distribution = Normal(self.mu, self.std)
        return self

    def entropy(self) -> torch.Tensor:
        """
        Get the entropy of the *base* Gaussian distribution.

        Note: This is an approximation. The true entropy of the squashed
        and scaled distribution is complex to compute. In many RL algorithms
        (like SAC), the "entropy bonus" is actually the expectation of the
        log-probability, not the true differential entropy.

        This returns the per-dimension entropy. The caller can .sum() if total
        entropy is needed.
        """
        # Get entropy from the base Normal distribution
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        """
        Sample an action, using the reparameterization trick (rsample).
        """
        # 1. Sample from Normal(mu, std) -> range (-inf, inf)
        z = self.distribution.rsample()

        # 2. Squash using tanh -> range [-1, 1]
        u = torch.tanh(z)

        # 3. Scale and shift to [low, high]
        return self.scale * u + self.bias

    def mode(self) -> torch.Tensor:
        """
        Return the mode of the distribution (most likely action).
        For a Gaussian, the mode is the mean (mu).
        We pass the mean through the deterministic transformations.
        """
        # 1. Mode of the base Gaussian is mu
        z = self.mu

        # 2. Squash using tanh
        u = torch.tanh(z)

        # 3. Scale and shift
        return self.scale * u + self.bias

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log-probability of taking a specific action.

        This requires inverting the transformations and applying the
        change of variables formula.

        :param actions: The actions taken (in range [low, high])
        """
        # --- Invert Transformations ---

        # 1. Un-scale from [low, high] back to [-1, 1]
        # u = (actions - bias) / scale
        u = (actions - self.bias) / self.scale

        # 2. Clamp for numerical stability (atanh is undefined at +/- 1)
        # We clamp *slightly* inside the [-1, 1] bounds.
        u = torch.clamp(u, -1.0 + self.epsilon, 1.0 - self.epsilon)

        # 3. Invert tanh to get the pre-squashed Gaussian sample (z)
        # z = atanh(u)
        z = 0.5 * torch.log((1 + u) / (1 - u))

        # 1. Get log-prob from the base Gaussian distribution
        # This has shape (batch_size, action_dim)
        log_prob_z = self.distribution.log_prob(z)

        # 2. Calculate the log-determinant of the Jacobian
        # log(1 - u^2)
        log_det_tanh = torch.log(1 - u.pow(2) + self.epsilon)

        # Total log-determinant (log_scale is scalar, broadcasts)
        # (batch_size, action_dim)
        log_det_jacobian = self.log_scale + log_det_tanh

        # 3. Apply the change of variables formula
        # log_prob(action) = log_prob(z) - log_det_jacobian
        log_prob = log_prob_z - log_det_jacobian

        # Returns log-prob per dimension, matching original class's behavior
        return log_prob.sum(dim=-1, keepdim=True)

    def sample_and_logprob(self):
        """
        Samples an action using the reparameterization trick AND computes
        its log-probability in a single, numerically stable pass.

        :return: (actions, log_prob)
        """
        # 1. Sample from Normal(mu, std) -> range (-inf, inf)
        z = self.distribution.rsample()

        # 2. Squash using tanh -> range [-1, 1]
        u = torch.tanh(z)

        # 3. Scale and shift to [low, high]
        actions = self.scale * u + self.bias

        # --- Calculate log_prob from z and u (numerically stable) ---

        # 1. Get log-prob from the base Gaussian distribution
        # This is log_prob(z)
        log_prob_z = self.distribution.log_prob(z)

        # 2. Calculate the log-determinant of the Jacobian
        # This is log( |da/dz| ) = log( |(scale * (1-u^2))| )
        # log(scale) + log(1 - u^2)
        log_det_tanh = torch.log(1 - u.pow(2) + self.epsilon)
        log_det_jacobian = self.log_scale + log_det_tanh

        # 3. Apply the change of variables formula
        # log_prob(action) = log_prob(z) - log_det_jacobian
        log_pi = log_prob_z - log_det_jacobian

        return actions, log_pi.sum(dim=-1, keepdim=True)


class SharedRecurrentEncoder(nn.Module):
    """
    This module encapsulates the shared parts of the network:
    1. The observation encoder (e.g., MiniGridEncoder)
    2. The LSTM layer

    It takes raw observations and outputs the
    sequence of LSTM hidden states (history-aware state features).
    """

    def __init__(self, input_dim, hidden_dim: int = 128,
                 recurrent_hidden_size: int = 128, device: str = 'cpu',
                 feature_extractor=None, burn_in_length: int = 0, *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        self.recurrent_hidden_size = recurrent_hidden_size

        if feature_extractor is None:
            # This encoder now processes individual frames
            self.encoder = FeatureEncoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
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

    def forward(self, x: Optional[torch.Tensor] = None,
                obs_features: Optional[torch.Tensor] = None,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                padding_mask: Optional[torch.Tensor] = None,
                train_mask: Optional[torch.Tensor] = None,
                lengths: Optional[List[int]] = None) -> Tuple[
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

        default_is_training_mask = torch.ones(
            lstm_input.shape[:2], dtype=torch.bool, device=lstm_input.device)

        if train_mask is not None:
            # --- Burn-in Logic with per-sequence length ---
            burn_in_lengths = self._get_burn_in_lengths(train_mask)  # shape: [B]

            # Create a mask of shape (B, T) to identify training steps.
            timesteps = torch.arange(seq_len, device=lstm_input.device).expand(len(burn_in_lengths), seq_len)
            is_training_mask_with_burn_in = timesteps >= burn_in_lengths.unsqueeze(1)

            is_any_training = ~(train_mask == padding_mask)
            has_training_steps_per_seq = torch.any(is_any_training, dim=1)

            is_training_mask = torch.where(
                has_training_steps_per_seq,
                is_training_mask_with_burn_in,
                default_is_training_mask
            )
        else:
            is_training_mask = default_is_training_mask

        # Use torch.where to create a new input tensor.
        # For training steps, use the original input to preserve the computation graph.
        # For burn-in steps, use a detached version to cut the graph.
        lstm_input = torch.where(
            is_training_mask.unsqueeze(-1),  # expand to [B, T, 1] to broadcast
            lstm_input,
            lstm_input.detach()
        )

        # Now, proceed with the masked input as if it were the original.
        packed_input = self._pack_sequence_maybe(lstm_input, padding_mask, lengths)
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
    def _pack_sequence_maybe(lstm_input, padding_mask, lengths=None):
        if padding_mask is not None:
            if lengths is None:
                lengths = padding_mask.sum(dim=1).squeeze(-1).cpu().to(torch.int64)
                lengths = torch.clamp(lengths, min=1)  # Ensure no zero lengths
            lstm_input = pack_padded_sequence(
                lstm_input,
                lengths,
                batch_first = True,
                enforce_sorted = False
            )

        return lstm_input

    @staticmethod
    def _unpack_sequence_maybe(lstm_out, seq_len):
        if isinstance(lstm_out, PackedSequence):
            lstm_out, _ = pad_packed_sequence(
                lstm_out,
                batch_first = True,
                total_length = seq_len
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
                 device: str = 'cpu', *args, **kwargs) -> None:
        super().__init__()
        self._device = device
        self._has_action_encoder = has_action_encoder

        if self._has_action_encoder:
            self.action_encoder = nn.Sequential(
                nn.Linear(1, input_feature_size // 2),
                nn.LayerNorm(input_feature_size // 2),
                nn.LeakyReLU(),
                nn.Linear(input_feature_size // 2, input_feature_size)
            )
            self.encoded_input_dim = input_feature_size * 2
        else:
            self.action_encoder = None
            self.encoded_input_dim = input_feature_size

        self.decoder = nn.Sequential(
            nn.Linear(self.encoded_input_dim, self.encoded_input_dim // 2),
            nn.LayerNorm(self.encoded_input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.encoded_input_dim // 2, output_size)
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
            input_features = torch.concat((input_features, encoded_actions), -1)

        batch_size, seq_len = input_features.shape[0], input_features.shape[1]

        # Reshape for the decoder
        # (N, T, input_feature_size) -> (N * T, input_feature_size)
        decoder_input = input_features.reshape(-1, self.encoded_input_dim)

        # (N * T, input_feature_size) -> (N * T, output_size)
        output = self.decoder(decoder_input)

        # Reshape final output back to sequence format
        # (N * T, output_size) -> (N, T, output_size)
        output = output.view(batch_size, seq_len, -1)

        return output


class _RecurrentBase(nn.Module):
    def __init__(self, observation_shape: Tuple[int,], hidden_dim: int = 128, gamma: float = 0.99,
                 recurrent_hidden_size: int = None, batch_size: int = 128,
                 device: str = 'cpu', critic_lr: float = 3e-4,
                 value_lr: float = 3e-4, actor_lr: float = 3e-4, sequence_length: int = 64,
                 burn_in_length: int = 20, seed: Optional[int] = None, *args, **kwargs):
        # Set our seed
        self._seed = seed
        if seed is not None:
            set_seed(seed)
        super().__init__()
        self._cloning_only = False
        self._device = device
        self.decoy_interval = None
        self.scaler = None
        self._scaler_dtype = None
        self._batch_diff = None
        self._critic_lr = critic_lr
        self._value_lr = value_lr
        self._actor_lr = actor_lr
        self._observation_shape = observation_shape
        self._hidden_dim = hidden_dim
        self._recurrent_hidden_size = self._hidden_dim if recurrent_hidden_size is None else recurrent_hidden_size
        self._batch_size = batch_size
        self.dist = CustomBetaDistribution(action_dim=1, low=INSULIN_ACTION_LOW, high=INSULIN_ACTION_HIGH).to(self._device)
        # self.dist = SquashedGaussianDistribution(action_dim=1, low=INSULIN_ACTION_LOW, high=INSULIN_ACTION_HIGH).to(self._device)
        self._eps = 1e-5
        self._tanh_scale = 1.0

        # --- PLACEHOLDERS: These must be set by the child class ---
        self.actor_encoder: SharedRecurrentEncoder = None
        self.policy_net: RecurrentNet = None
        # ---

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
            decoy_interval: int = 0,
            dataset_kwargs: Optional[Dict] = None
    ):
        self.decoy_interval = decoy_interval
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
            self._scaler_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Initialise our dataset and loss dictionary
        dataset_kwargs = dict() if dataset_kwargs is None else dataset_kwargs
        dataset.set_generate_params(self._device, max_sequence_length=self._sequence_length,
                                    decoy_interval=decoy_interval, burn_in_length=self._burn_in_length,
                                    batch_size=self._batch_size, **dataset_kwargs)

        loss_dict = self._reset_loss_dict()

        # Start training
        with tqdm(total=n_epochs_train * len(dataset), desc="Progress", mininterval=2.0,
                  disable=not show_progress) as pbar:
            for epoch in range(1, n_epochs_train + 1):
                epoch_str = f"{epoch}/{n_epochs_train}"

                for batch in dataset:

                    # Update the networks
                    if not self._cloning_only:
                        loss_dict['critic_loss'].append(self.update_critic(*batch))
                        loss_dict['value_loss'].append(self.update_value(*batch))
                    loss_dict['policy_loss'].append(self.update_actor(*batch))

                    if self.scaler is not None:
                        self.scaler.update()

                    # Soft update of target value network
                    self.sync_target_networks()

                    pbar.update(1)
                    pbar.set_postfix(epoch=epoch_str,
                                     policy_loss=f"{np.mean(loss_dict['policy_loss']):.5f}",
                                     critic_loss=f"{np.mean(loss_dict['critic_loss']):.5f}",
                                     value_loss=f"{np.mean(loss_dict['value_loss']):.5f}",
                                     refresh=False)

                # Logging
                if epoch < n_epochs_train and epoch % n_epochs_per_eval == 0 and evaluators is not None:
                    loss_dict, _ = self._log_progress(
                        epoch=epoch,
                        loss_dict=loss_dict,
                        evaluators=evaluators
                    )

            _, log_dict = self._log_progress(
                epoch=epoch,
                loss_dict=loss_dict,
                evaluators=evaluators
            )

        return log_dict

    def update_critic(self, *args):
        return torch.nan

    def update_value(self, *args):
        return torch.nan

    def update_actor(self, *args):
        return torch.nan

    def generic_update(self, loss, optimizer, *nets):
        optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            for net in nets:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            self.scaler.step(optimizer)
        else:
            loss.backward()
            for net in nets:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
        return loss.item()

    def sync_target_networks(self):
        pass

    def predict(self, obs: np.ndarray, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                deterministic: bool = False, with_dist: bool = False) -> Tuple[Optional[CustomBetaDistribution], np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predicts an action for a single observation and manages the recurrent state.
        Returns the action and the next hidden state.

        Expects obs to be of the relevant shape already (N, L, 1)
        """
        obs_tensor = self._to_tensors(obs)[0]
        assert obs_tensor.ndim == 3, "Observation must have shape (N, L, 1)."

        # Pass the current hidden_state to the policy network
        # 1. Call the shared encoder
        #    Input: (N, L, obs), Output: (N, L, D)
        assert self.actor_encoder is not None, "Missing actor encoder in the recurrent network."
        lstm_out_pi, next_hidden_state = self.actor_encoder(obs_tensor,
                                                            hidden_state=hidden_state,
                                                            padding_mask=None,
                                                            train_mask=None,
                                                            lengths=None)

        # 2. Call the policy decoder
        #    Input: (1, 1, hidden_size), Output: (1, 1, 2)
        assert self.policy_net is not None, "Missing policy network in the recurrent network."
        params = self.policy_net(lstm_out_pi, actions=None)

        dist = self.dist.proba_distribution(params)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if with_dist:
            return dist, action.squeeze(-1).cpu().numpy(), next_hidden_state

        return action.squeeze(-1).cpu().numpy(), next_hidden_state

    def evaluate_actions(self, obs, hidden_state, actions):
        dist, _, _ = self.predict(obs, hidden_state=hidden_state, deterministic=True, with_dist=True)
        log_probs = dist.log_prob(actions)
        return log_probs

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

        def next_true_idx(_mask):  # earliest j >= t if exists else T
            cand = torch.where(_mask, t_idx, none)
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

    def _log_progress(self, epoch: int, loss_dict: dict, evaluators: dict = None):
        log_rewards = {}
        eval_str = '\n' + '=' * 40 + f"\nEpoch {epoch}:"
        for key in evaluators.keys():
            eval_output = evaluators[key](self, seed=self._seed)

            if isinstance(eval_output, float):
                # WIS estimate
                log_rewards[key] = eval_output
                eval_str += f"\n     {key} = {eval_output:.2f} (WIS discounted estimate)"
                continue

            episodic_rewards, discounted_episodic_rewards = eval_output

            episodic_rewards = np.array(episodic_rewards)
            discounted_episodic_rewards = np.array(discounted_episodic_rewards)
            log_rewards[key] = episodic_rewards.mean()
            log_rewards[key + '_discounted'] = discounted_episodic_rewards.mean()

            # Get IQM for printout only
            for arr_key, arr in [('', episodic_rewards), ('_discounted', discounted_episodic_rewards)]:
                iqr = trimboth(arr, proportiontocut=0.25)
                iqr_mean = np.mean(iqr)
                iqr_std = np.std(iqr)
                iqr_n_samples = len(iqr)

                eval_str += f"\n     {key + arr_key} = {iqr_mean:.2f} +/- {iqr_std / np.sqrt(iqr_n_samples):.2f}"

        eval_str += f"\n\n     policy_loss = {np.mean(loss_dict['policy_loss']):.7f}"
        eval_str += f"\n     critic_loss = {np.mean(loss_dict['critic_loss']):.7f}"
        eval_str += f"\n     value_loss = {np.mean(loss_dict['value_loss']):.7f}\n"
        eval_str += '=' * 40 + '\n'
        print(eval_str)

        return self._reset_loss_dict(), log_rewards

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
            beta: float = 2.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._expectile = expectile
        self._cloning_only = expectile == 0.5
        self._tau_target = tau_target
        self._beta = beta

        # Kwargs for encoders and decoders
        encoder_kwargs = dict(
            input_dim=self._observation_shape[0],
            hidden_dim=self._hidden_dim,
            recurrent_hidden_size=self._recurrent_hidden_size,
            device=self._device
        )

        decoder_kwargs = dict(
            input_feature_size=self._recurrent_hidden_size,
            device=self._device
        )

        # --- Create Shared Encoders ---
        self.shared_critic_encoder = SharedRecurrentEncoder(
            feature_extractor=FeatureEncoder(**encoder_kwargs).to(self._device),
            **encoder_kwargs
        ).to(self._device)
        self.target_shared_critic_encoder = SharedRecurrentEncoder(
            feature_extractor=FeatureEncoder(**encoder_kwargs).to(self._device),
            **encoder_kwargs
        ).to(self._device)
        self.actor_encoder = SharedRecurrentEncoder(
            feature_extractor=FeatureEncoder(**encoder_kwargs),
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
        self.critic_optim = torch.optim.Adam(
            list(self.shared_critic_encoder.parameters())
            + list(self.critic_net1.parameters())
            + list(self.critic_net2.parameters()),
            lr=self._critic_lr
        )
        self.value_optim = torch.optim.Adam(
            list(self.shared_critic_encoder.parameters())
            + list(self.value_net.parameters()),
            lr=self._value_lr
        )
        self.policy_optim = torch.optim.Adam(
            list(self.actor_encoder.parameters())
            + list(self.policy_net.parameters()),
            lr=self._actor_lr
        )

        # clone the value net to a target network
        self.sync_target_networks(tau=1.0)

    def forward(self, obs, acts):
        # Note that this returns both the output and the hidden states
        lstm_out_q, _ = self.shared_critic_encoder(obs)
        lstm_out_pi, _ = self.actor_encoder(obs)
        q1 = self.critic_net1(lstm_out_q, actions=acts)
        q2 = self.critic_net2(lstm_out_q, actions=acts)

        v = self.value_net(lstm_out_q)
        logits = self.policy_net(lstm_out_pi)

        return (q1, q2), v, logits

    def sync_target_networks(self, tau=None):
        tau = tau or self._tau_target
        # Sync the target value decoder
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

        # Sync the target shared encoder
        for target_param, param in zip(self.target_shared_critic_encoder.parameters(),
                                       self.shared_critic_encoder.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    def update_critic(self, *batch):
        # Scale the loss and perform the backward pass
        critic_loss = self._update_critic(*batch)
        critic_loss = self.generic_update(
            critic_loss,
            self.critic_optim,
            self.critic_net1, self.critic_net2, self.shared_critic_encoder
        )

        return critic_loss

    @torch.compile(mode='default') # reduce-overhead leads to NaNs here
    def _update_critic_compiled(self, obs, acts, next_obs, padding_mask, next_padding_mask, train_mask, lengths,
                                next_lengths):
        # We only need the network output, not the final hidden state for training
        lstm_out_q, hidden_state = self.shared_critic_encoder(obs, padding_mask=padding_mask, train_mask=train_mask,
                                                              lengths=lengths)
        q1 = self.critic_net1(lstm_out_q, actions=acts)
        q2 = self.critic_net2(lstm_out_q, actions=acts)

        with torch.no_grad():
            # Use target shared encoder and target value decoder
            next_lstm_out_tgt, _ = self.target_shared_critic_encoder(next_obs, padding_mask=next_padding_mask,
                                                                     train_mask=None, lengths=next_lengths)
            v_next = self.target_value_net(next_lstm_out_tgt, actions=None)

        return q1, q2, v_next

    def _update_critic(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                       train_mask, lengths, next_lengths):
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            q1, q2, v_next = self._update_critic_compiled(obs, acts, next_obs, padding_mask, next_padding_mask,
                                                          train_mask, lengths, next_lengths)

            if self.decoy_interval == 0:
                q1, q2, q_target, _ = self._filter_to_correct_visibles(
                    q1, q2, rews, v_next, dones, visible, next_visible, padding_mask, train_mask)

                critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            else:
                q_target = rews + self._gamma * (1 - dones.float()) * v_next

                q1_loss = F.mse_loss(q1, q_target, reduction='none')
                q2_loss = F.mse_loss(q2, q_target, reduction='none')
                critic_loss_unmasked = q1_loss + q2_loss

                # Apply the mask
                masked_loss = critic_loss_unmasked * train_mask
                critic_loss = masked_loss.sum() / (train_mask.sum() + self._eps)

        return critic_loss

    def update_value(self, *batch):
        # Scale the loss and perform the backward pass
        value_loss, diff = self._update_value(*batch)
        value_loss = self.generic_update(
            value_loss,
            self.value_optim,
            self.value_net, self.shared_critic_encoder
        )

        self._batch_diff = diff.clone()

        return value_loss

    @torch.compile(mode='reduce-overhead')
    def _update_value(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                      train_mask, lengths, next_lengths):
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            # Value function doesn't depend on actions
            lstm_out_q, _ = self.shared_critic_encoder(obs, padding_mask=padding_mask, train_mask=train_mask,
                                                       lengths=lengths)
            v = self.value_net(lstm_out_q, actions=None)

            with torch.no_grad():
                q1 = self.critic_net1(lstm_out_q, actions=acts)
                q2 = self.critic_net2(lstm_out_q, actions=acts)
                q = torch.min(q1, q2)

            diff = q - v
            weights = torch.absolute(self._expectile - (diff < 0).float())
            value_loss_unmasked = (weights * (diff.float() ** 2))  # Use .float() to avoid bfloat16 issues

            # Apply the mask
            final_mask = train_mask & visible
            masked_loss = value_loss_unmasked * final_mask
            value_loss = masked_loss.sum() / (final_mask.sum() + self._eps)

        return value_loss, diff.detach()

    def update_actor(self, *batch):
        # Scale the loss and perform the backward pass
        diff = self._batch_diff.clone() if self._batch_diff is not None else None
        policy_loss, variance_loss = self._update_actor(*batch, diff)
        total_loss = policy_loss + variance_loss
        _ = self.generic_update(
            total_loss,
            self.policy_optim,
            self.policy_net, self.actor_encoder
        )

        return policy_loss.item()

    @torch.compile(mode='default')
    def _update_actor(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                      train_mask, lengths, next_lengths, diff):
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            # Policy doesn't depend on actions
            lstm_out_pi, _ = self.actor_encoder(obs, padding_mask=padding_mask, train_mask=train_mask,
                                                lengths=lengths)
            params = self.policy_net(lstm_out_pi, actions=None)

            dist = self.dist.proba_distribution(params)

            # Calculate the log-probability of the expert actions `acts`
            log_pi = dist.log_prob(acts)

            # The policy loss is the Negative Log-Likelihood (NLL)
            policy_loss_unmasked = -log_pi

            # Get the variance target
            variance = dist.variance()
            variance_loss = (variance - 0.001) ** 2

            # Get the IQL weights
            weights = 1.0
            if not self._cloning_only:
                weights = torch.clip(torch.exp(self._beta * diff), 0, 100)

            # Apply the mask
            final_mask = train_mask & visible
            masked_loss = policy_loss_unmasked * weights * final_mask
            policy_loss = masked_loss.sum() / (final_mask.sum() + self._eps)

            variance_loss = (variance_loss * weights * final_mask).sum() / (final_mask.sum() + self._eps)
            # Scale to match policy loss
            variance_loss = (policy_loss.abs() / variance_loss.abs()).detach() * variance_loss

        return policy_loss, variance_loss


class RecurrentCQLSAC(_RecurrentBase):
    def __init__(
            self,
            tau_target: float = 0.005,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._tau_target = tau_target
        self._target_entropy_alpha = -1
        self._target_cql_alpha_gap = 0.
        self._entropy_alpha = torch.tensor(1.0, device=self._device)
        self._cql_alpha = torch.tensor(1.0, device=self._device)
        self.cql_uniform_log_probs = -torch.log(torch.tensor(INSULIN_ACTION_HIGH - INSULIN_ACTION_LOW,
                                                             device=self._device))

        # Kwargs for encoders and decoders
        encoder_kwargs = dict(
            input_dim=self._observation_shape[0],
            hidden_dim=self._hidden_dim,
            recurrent_hidden_size=self._recurrent_hidden_size,
            device=self._device
        )

        decoder_kwargs = dict(
            input_feature_size=self._recurrent_hidden_size,
            device=self._device
        )

        # --- Create Shared Encoders ---
        self.shared_critic_encoder = SharedRecurrentEncoder(
            feature_extractor=FeatureEncoder(**encoder_kwargs).to(self._device),
            **encoder_kwargs
        ).to(self._device)
        self.target_shared_critic_encoder = SharedRecurrentEncoder(
            feature_extractor=FeatureEncoder(**encoder_kwargs).to(self._device),
            **encoder_kwargs
        ).to(self._device)
        self.actor_encoder = SharedRecurrentEncoder(
            feature_extractor=FeatureEncoder(**encoder_kwargs),
            **encoder_kwargs
        ).to(self._device)

        # --- Create Decoders ---
        self.critic_net1 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)
        self.critic_net2 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)
        self.target_critic_net1 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(
            self._device)
        self.target_critic_net2 = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(
            self._device)
        self.policy_net = RecurrentNet(output_size=2, **decoder_kwargs).to(self._device)

        # --- Setup Optimizers ---
        self.critic_optim = torch.optim.Adam(
            list(self.shared_critic_encoder.parameters())
            + list(self.critic_net1.parameters())
            + list(self.critic_net2.parameters()),
            lr=self._critic_lr
        )
        self.policy_optim = torch.optim.Adam(
            list(self.actor_encoder.parameters())
            + list(self.policy_net.parameters()),
            lr=self._actor_lr
        )

        # Initialize target networks
        self.sync_target_networks(tau=1.0)

    def forward(self, obs, acts):
        # Note that this returns both the output and the hidden states
        lstm_out_q, _ = self.shared_critic_encoder(obs)
        lstm_out_pi, _ = self.actor_encoder(obs)

        q1_logits = self.critic_net1(lstm_out_q, actions=acts)
        q2_logits = self.critic_net2(lstm_out_q, actions=acts)
        policy_logits = self.policy_net(lstm_out_pi)

        return q1_logits, q2_logits, policy_logits

    def update_critic(self, *batch):
        # Scale the loss and perform the backward pass
        critic_loss = self._update_critic(*batch)
        critic_loss = self.generic_update(
            critic_loss,
            self.critic_optim,
            self.critic_net1, self.critic_net2, self.shared_critic_encoder
        )
        return critic_loss

    @torch.compile(mode='default')
    def _update_critic_inference_compiled(self, obs, acts, next_obs, visible, padding_mask, next_padding_mask,
                                          train_mask, lengths, next_lengths):
        # -- Inference for Bellman loss -- #
        lstm_out_q, _ = self.shared_critic_encoder(obs, padding_mask=padding_mask, train_mask=train_mask,
                                                   lengths=lengths)
        q1_pred = self.critic_net1(lstm_out_q, actions=acts)
        q2_pred = self.critic_net2(lstm_out_q, actions=acts)

        with torch.no_grad():
            # 2. Get our a' sampled from our policy
            next_lstm_out_pi, _ = self.actor_encoder(next_obs, padding_mask=next_padding_mask, train_mask=None,
                                                     lengths=next_lengths)
            params = self.policy_net(next_lstm_out_pi, actions=None)

            # Create the Beta distribution
            dist = self.dist.proba_distribution(params)

            # Sample next action
            next_actions_sampled = dist.sample()
            next_log_pi = dist.log_prob(next_actions_sampled)

            # 3. Get our Q(s',a') using the sampled action
            next_lstm_out_tgt, _ = self.target_shared_critic_encoder(next_obs, padding_mask=next_padding_mask,
                                                                     train_mask=None, lengths=next_lengths)
            next_q1 = self.target_critic_net1(next_lstm_out_tgt, actions=next_actions_sampled)
            next_q2 = self.target_critic_net2(next_lstm_out_tgt, actions=next_actions_sampled)
            next_q = torch.min(next_q1, next_q2)

            # Add entropy regularisation
            next_v = next_q - self._entropy_alpha * next_log_pi

        # -- Inference for CQL loss -- #
        batch_size, seq_len, obs_dim = obs.shape
        cql_n_samples = 10  # for each of policy and uniform
        collapsed_dim = (cql_n_samples * 2 * batch_size, seq_len, -1)
        expanded_dim = (cql_n_samples * 2, batch_size, seq_len, -1)
        cql_lstm_out_q = (
            lstm_out_q.unsqueeze(0)
            .expand(cql_n_samples * 2, batch_size, seq_len, -1)
            .contiguous()
            .view(*collapsed_dim)
        )

        with torch.no_grad():
            # 5. Sample our actions
            cql_lstm_out_pi, _ = self.actor_encoder(obs, padding_mask=padding_mask, train_mask=None,
                                                    lengths=lengths)
            params = self.policy_net(cql_lstm_out_pi, actions=None)

            # Create the Beta distribution
            dist = self.dist.proba_distribution(params)

            # Sample policy and uniform actions -> [5, N, L, D]
            cql_policy_actions = torch.stack([dist.sample() for _ in range(cql_n_samples)])
            cql_uniform_actions = torch.empty(cql_n_samples, batch_size, seq_len, 1, device=obs.device)
            cql_uniform_actions.uniform_(INSULIN_ACTION_LOW, INSULIN_ACTION_HIGH)

            # Get the log_probs for these actions
            cql_policy_log_probs = dist.log_prob(cql_policy_actions)
            cql_uniform_log_probs = self.cql_uniform_log_probs.expand_as(cql_policy_log_probs)

            # Concat together and reshape back to N,L,D
            cql_sampled_actions = torch.cat([cql_policy_actions, cql_uniform_actions], 0)
            cql_sampled_log_probs = torch.cat([cql_policy_log_probs, cql_uniform_log_probs], 0)

            cql_sampled_actions = cql_sampled_actions.view(*collapsed_dim)
            cql_sampled_log_probs = cql_sampled_log_probs.view(*collapsed_dim)

        # Calculate the Q values for these sampled actions -> [10, N, L]
        cql_q1_sampled = self.critic_net1(cql_lstm_out_q, actions=cql_sampled_actions) - cql_sampled_log_probs
        cql_q2_sampled = self.critic_net2(cql_lstm_out_q, actions=cql_sampled_actions) - cql_sampled_log_probs

        # Calculate the logsumexp
        temperature = 1.0
        cql_q1_sampled = cql_q1_sampled / temperature
        cql_q2_sampled = cql_q2_sampled / temperature
        cql_q1_logsumexp = torch.logsumexp(cql_q1_sampled.view(*expanded_dim), dim=0) * temperature - np.log(
            2 * cql_n_samples)
        cql_q2_logsumexp = torch.logsumexp(cql_q2_sampled.view(*expanded_dim), dim=0) * temperature - np.log(
            2 * cql_n_samples)

        # Subtract the mean dataset Q from the logsumexp Q
        cql_loss1 = cql_q1_logsumexp - q1_pred
        cql_loss2 = cql_q2_logsumexp - q2_pred

        # Apply and update our alpha penalty
        scaled_cql_loss1 = self._cql_alpha * (cql_loss1 - self._target_cql_alpha_gap)
        scaled_cql_loss2 = self._cql_alpha * (cql_loss2 - self._target_cql_alpha_gap)
        scaled_cql_loss = scaled_cql_loss1 + scaled_cql_loss2

        # Use train_mask (which is False for padding AND burn-in)
        final_mask = train_mask * visible  # Shape (N, T)
        valid_count = final_mask.sum() + self._eps

        scaled_cql_loss = scaled_cql_loss * final_mask
        scaled_cql_loss = scaled_cql_loss.sum() / valid_count

        return q1_pred, q2_pred, next_v, scaled_cql_loss

    def _update_critic(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                       train_mask, lengths, next_lengths):
        '''
        CQL Q-loss: logsumexp_loss + bellman_error_loss
        1) bellman_error_loss = MSE(Q(s,a), r + gamma * (1 - done) * V(s'))
        2) logsumexp_loss = logsumexp(Q(s,a')) - E_{a~D}[Q(s,a)]

        bellman_error_loss
        1. Get the predicted Q-values for the observed state-action pair
        2. Get our a' sampled from our policy
        3. Get our Q(s',a') using the sampled a'
        4. Create the Bellman loss

        logsumexp_loss
        5. Sample N actions - five from our policy, five from a uniform distribution (to cover the entire
            action space)
        6. Calculate the Q values for these sampled actions
        7. Calculate the logsumexp for these Q values
        8. Subtract the mean dataset Q from the logsumexp Q
        9. Update our Lagrangian alpha
        '''
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            q1_pred, q2_pred, next_v, cql_loss = self._update_critic_inference_compiled(
                obs, acts, next_obs, visible, padding_mask, next_padding_mask, train_mask, lengths, next_lengths
            )

            # Create the Bellman loss
            if self.decoy_interval == 0:
                q1_pred, q2_pred, q_target, valid_mask = self._filter_to_correct_visibles(
                    q1_pred, q2_pred, rews, next_v, dones, visible, next_visible, padding_mask, train_mask)

                bellman_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

            else:
                q_target = rews + self._gamma * (1 - dones.float()) * next_v

                q1_loss = F.mse_loss(q1_pred, q_target, reduction='none')
                q2_loss = F.mse_loss(q2_pred, q_target, reduction='none')

                bellman_loss = q1_loss + q2_loss  # Shape [N, T, 1]

                # Use train_mask (which is False for padding AND burn-in)
                valid_count = train_mask.sum() + self._eps
                bellman_loss = (bellman_loss * train_mask)
                bellman_loss = bellman_loss.sum() / valid_count

            critic_loss = bellman_loss + cql_loss

        return critic_loss

    def update_actor(self, *batch):
        # Scale the loss and perform the backward pass
        policy_loss = self._update_actor(*batch)
        policy_loss = self.generic_update(
            policy_loss,
            self.policy_optim,
            self.policy_net, self.actor_encoder
        )
        return policy_loss

    @torch.compile(mode='reduce-overhead')
    def _update_actor(self, obs, acts, rews, next_obs, dones, visible, next_visible, padding_mask, next_padding_mask,
                      train_mask, lengths, next_lengths):
        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            # 1. Get policy params (action-independent)
            lstm_out_pi, _ = self.actor_encoder(obs, padding_mask=padding_mask, train_mask=train_mask,
                                                lengths=lengths)
            params = self.policy_net(lstm_out_pi, actions=None)

            # Create the Beta distribution
            dist = self.dist.proba_distribution(params)

            # Sample an action (rsample) in [low, high] range
            actions_sampled, log_pi_sampled = dist.sample_and_logprob()

            # Get Q-values for these new actions
            # We pass u_sampled (normalized action) to the critics, as they expect.
            # We do *not* detach gradients here, as we need them to flow through Q into u_sampled
            with torch.no_grad():
                lstm_out_q, _ = self.shared_critic_encoder(obs, padding_mask=padding_mask, train_mask=None,
                                                           lengths=lengths)
                q1_dataset = self.critic_net1(lstm_out_q, actions=acts)
                q2_dataset = self.critic_net2(lstm_out_q, actions=acts)
                q_values = torch.min(q1_dataset, q2_dataset)

            q1_sampled = self.critic_net1(lstm_out_q, actions=actions_sampled)
            q2_sampled = self.critic_net2(lstm_out_q, actions=actions_sampled)
            q_sampled = torch.min(q1_sampled, q2_sampled)

            # Calculate SAC Actor Loss
            # The loss is (alpha * log_prob) - Q_value
            # Having (q_sampled - q_values) rather than q_sampled on its own doesn't change the gradient,
            # but does tell us if the actor is converging to positive-advantage actions during loss inspection.
            policy_loss = (self._entropy_alpha * log_pi_sampled - (q_sampled - q_values))

            # Apply the mask to our losses
            final_mask = (train_mask * visible)
            policy_loss = (policy_loss * final_mask)
            policy_loss = policy_loss.sum() / (final_mask.sum() + self._eps)

        return policy_loss

    def sync_target_networks(self, tau=None):
        tau = tau or self._tau_target
        # Sync target critic decoders
        for target_param, param in zip(self.target_critic_net1.parameters(), self.critic_net1.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
        for target_param, param in zip(self.target_critic_net2.parameters(), self.critic_net2.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

        # Sync target shared encoder
        for target_param, param in zip(self.target_shared_critic_encoder.parameters(),
                                       self.shared_critic_encoder.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
