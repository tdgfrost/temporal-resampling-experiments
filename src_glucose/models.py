from collections import deque, defaultdict
from functools import partial
from copy import deepcopy
from typing import Tuple, Dict, Optional, Union, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import trimboth
from torch.distributions import Beta, Normal
from tqdm import tqdm

from gym_wrappers import INSULIN_ACTION_LOW, INSULIN_ACTION_HIGH
from ppo_trainer import set_seed, RecurrentPPO

torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.backends.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"
torch.backends.cudnn.benchmark = True


class CallableDummyDist(nn.Module):
    def __init__(self, use_dataset: bool = False):
        super().__init__()
        self.params = None
        self.use_dataset = use_dataset

    def proba_distribution(self, params: torch.Tensor, *args, **kwargs):
        self.params = params
        return self

    def sample(self, size: Tuple = (1,), *args, **kwargs):
        (batch_size,) = size  # Must be tuple of length 1
        if batch_size > 1:
            self.params = self.params.new_empty(batch_size, *self.params.shape)
        self.params.uniform_(INSULIN_ACTION_LOW, INSULIN_ACTION_HIGH)
        return self.params

    def mode(self, *args, **kwargs):
        if self.use_dataset:
            return None
        self.params.uniform_(INSULIN_ACTION_LOW, INSULIN_ACTION_HIGH)
        return self.params


class CallableRandomAgentForFQE(nn.Module):
    def __init__(self, use_dataset: bool = False, *args, **kwargs):
        super().__init__()
        self.dist = CallableDummyDist(use_dataset=use_dataset)
        self.use_dataset = use_dataset

    def predict(self, obs, deterministic: bool = False, *args, **kwargs):
        params = self.policy_net(obs)
        self.dist.proba_distribution(params)
        if deterministic:
            actions = self.dist.mode()
        else:
            actions = self.dist.sample()
        return actions, None

    @staticmethod
    def actor_encoder(obs, *args, **kwargs):
        return obs, None

    def policy_net(self, obs, *args, **kwargs):
        if self.use_dataset:
            return None
        batch_size, seq_len, _ = obs.shape
        return torch.empty(batch_size, seq_len, 1, device=obs.device)


class CallablePPOAgentForFQE(nn.Module):
    def __init__(self, ppo_agent: RecurrentPPO, *args, **kwargs):
        super().__init__()
        self.ppo_agent = ppo_agent
        self.dist = self.ppo_agent.ac_network.dist

    def predict(self, *args, **kwargs):
        return self.ppo_agent.predict(*args, **kwargs)

    def actor_encoder(self, obs, *args, **kwargs):
        lstm_out, _, _ = self.ppo_agent.ac_network.forward_lstm(obs)
        return lstm_out, None

    def policy_net(self, lstm_out, *args, **kwargs):
        actor_out = self.ppo_agent.ac_network.actor_head(lstm_out)
        return actor_out


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
        alpha_logits, beta_logits = torch.chunk(params, 2, dim=-1)

        self.alpha = F.softplus(alpha_logits) + 1.0
        self.beta = F.softplus(beta_logits) + 1.0

        self.mu = self.alpha / (self.alpha + self.beta)
        self.kappa = self.alpha + self.beta

        # (optional) Clamp for numerical stability
        # self.alpha = torch.clamp(self.alpha, 1.0, self.clamp_max)
        # self.beta = torch.clamp(self.beta, 1.0, self.clamp_max)

        # Create the underlying Beta distribution
        self.distribution = Beta(self.alpha, self.beta, validate_args=False)
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

    def sample(self, size: Tuple = (1,)) -> torch.Tensor:
        """
        Sample an action, using the reparameterization trick (rsample).
        """
        # Sample from Beta(alpha, beta) -> range [0, 1]
        u = self.distribution.rsample(size).squeeze(0)  # If size is 1, remove extra dim

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
        return log_prob

    def sample_and_logprob(self, size: Iterable = (1,)):
        """
        Samples an action using the reparameterization trick AND computes
        its log-probability in a single, numerically stable pass.

        :return: (actions, log_prob)
        """
        # 1. Sample from Beta(alpha, beta) -> range [0, 1]
        u = self.distribution.rsample(size).squeeze(0)  # If size is 1, remove extra dim

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

        return actions, log_pi

    def variance(self):
        mu = self.mu.detach()
        variance = (mu * (1 - mu)) / (self.kappa + 1)
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
        self.distribution = Normal(self.mu, self.std, validate_args=False)
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

        if train_mask is not None:
            is_training_mask = train_mask
        else:
            is_training_mask = torch.ones(
                lstm_input.shape[:2], dtype=torch.bool, device=lstm_input.device).unsqueeze(-1)

        # Use torch.where to create a new input tensor.
        # For training steps, use the original input to preserve the computation graph.
        # For burn-in steps, use a detached version to cut the graph.
        lstm_input = torch.where(
            is_training_mask,  # expand to [B, T, 1] to broadcast
            lstm_input,
            lstm_input.detach()
        )

        # Now, proceed with the masked input as if it were the original.
        lstm_out, next_hidden_state = self.lstm(lstm_input, hidden_state)

        return lstm_out, next_hidden_state


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

        batch_size, seq_len = input_features.shape[0], input_features.shape[1]

        if actions is not None:
            assert actions.size(0) == input_features.size(0), \
            f"Batch size mismatch between input_features and actions - {input_features.size(0)} vs {actions.size(0)}."
            assert seq_len == 1 or actions.size(1) == input_features.size(1), \
            f"Sequence length mismatch between input_features and actions - {input_features.size(1)} vs {actions.size(1)}."
            while actions.ndim < input_features.ndim:
                actions = actions.unsqueeze(-1)

            encoded_actions = self.action_encoder(actions)
            input_features = torch.concat((input_features, encoded_actions), -1)

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
        self._best_model_state = None
        self._cloning_only = False
        self._device = device
        self.decoy_interval = None
        self.scaler = None
        self._scaler_dtype = None
        self._batch_diff = None
        self.steps_per_epoch = None
        self._critic_lr = critic_lr
        self._value_lr = value_lr
        self._actor_lr = actor_lr
        self._observation_shape = observation_shape
        self._hidden_dim = hidden_dim
        self._recurrent_hidden_size = self._hidden_dim if recurrent_hidden_size is None else recurrent_hidden_size
        self._batch_size = batch_size
        self.dist = CustomBetaDistribution(action_dim=1, low=INSULIN_ACTION_LOW, high=INSULIN_ACTION_HIGH).to(
            self._device)
        self._eps = 1e-5
        self._tanh_scale = 1.0

        # --- PLACEHOLDERS: These must be set by the child class ---
        self.actor_encoder: SharedRecurrentEncoder = None
        self.policy_net: RecurrentNet = None

        self.update_funcs = dict()
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
            evaluators_val=None,
            evaluators_test=None,
            early_stopping_key: Optional[str] = None,
            show_progress: bool = True,
            decoy_interval: int = 0,
            early_stopping_limit: Optional[int] = 3,
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

        dataloader = iter(dataset)

        # Set up our evaluators
        assert not (evaluators is not None and (evaluators_val is not None or evaluators_test is not None)), \
            "Please specify either 'evaluators' OR 'evaluators_val/evaluators_test', not both."
        evaluators_val = evaluators_val or evaluators
        evaluators_test = evaluators_test or evaluators
        assert early_stopping_limit == 0 or early_stopping_key is not None, \
            "Please specify 'early_stopping_key' if using early stopping."

        loss_dict = self._reset_loss_dict()
        stop_early_count = 0
        best_online_return = -1 * float('inf')
        best_log_dict = None
        best_epoch = None

        assert self.steps_per_epoch is not None, \
            "Please specify self.steps_per_epoch in the child class."

        steps_per_epoch = self.steps_per_epoch[decoy_interval]

        # Start training
        total_steps = n_epochs_train * steps_per_epoch
        with tqdm(total=total_steps, desc="Progress", mininterval=2.0,
                  disable=not show_progress) as pbar:
            for epoch in range(1, n_epochs_train + 1):
                epoch_str = f"{epoch}/{n_epochs_train}"

                for step in range(steps_per_epoch):
                    # Pull next batch from infinite stream
                    batch = next(dataloader)

                    # Update the networks
                    for key, func in self.update_funcs.items():
                        loss_dict[key].append(func(*batch))

                    if self.scaler is not None:
                        self.scaler.update()

                    # Soft update of target value network
                    self.sync_target_networks()

                    pbar.update(1)
                    if (step + 1) % 50 == 0:
                        pbar_dict = {'epoch': epoch_str,
                                     'refresh': False}
                        for key in loss_dict.keys():
                            pbar_dict.update({key: f"{torch.stack(list(loss_dict[key])).mean().item():.5f}"})

                        pbar.set_postfix(**pbar_dict)

                # Logging
                if epoch < n_epochs_train:
                    do_eval = (epoch % n_epochs_per_eval == 0) and (evaluators_val is not None)
                    if do_eval:
                        log_dict = self._log_progress(
                            epoch=epoch,
                            evaluators=evaluators_val
                        )

                        current_return = log_dict[early_stopping_key]
                        if current_return > best_online_return:
                            best_online_return = current_return
                            best_log_dict = log_dict
                            best_epoch = epoch
                            stop_early_count = 0
                            self._save_best_model_state()
                        else:
                            stop_early_count += 1

                        if early_stopping_limit is not None and stop_early_count >= early_stopping_limit:
                            print(f"Early stopping triggered at epoch {epoch} "
                                  f"- loading previous best state at epoch {best_epoch}.")
                            self._load_best_model_state()
                            break

            if (best_log_dict is None) or (evaluators_test is not evaluators_val):
                best_log_dict = self._log_progress(
                    epoch="final",
                    evaluators=evaluators_test
                )

        return best_log_dict

    @staticmethod
    def add_noise(obs):
        noise = torch.randn_like(obs) * obs.view(-1, 4).std(0)
        new_obs = obs.clone()  # necessary for cudagraphs
        new_obs[..., :-1] = new_obs[..., :-1] + noise[..., :-1] * 0.01
        return new_obs

    def generic_update(self, loss, optimizer, *nets):
        optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            for net in nets:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            self.scaler.step(optimizer)
        else:
            loss.backward()
            for net in nets:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        return loss.detach().clone()

    def sync_target_networks(self):
        pass

    def predict(self, obs: np.ndarray, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                deterministic: bool = False, with_dist: bool = False, action_as_tensor: bool = False,
                *args, **kwargs) -> Union[
        Tuple[CustomBetaDistribution, np.ndarray, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]
    ]:
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
                                                            train_mask=None)

        # 2. Call the policy decoder
        #    Input: (1, 1, hidden_size), Output: (1, 1, 2)
        assert self.policy_net is not None, "Missing policy network in the recurrent network."
        params = self.policy_net(lstm_out_pi, actions=None)

        dist = self.dist.proba_distribution(params)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if not action_as_tensor:
            action = action.detach().flatten().cpu().numpy()

        if with_dist:
            return dist, action, next_hidden_state

        return action, next_hidden_state

    @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def _filter_to_correct_visibles(self, r, v_next, dones, visible, next_visible, padding_mask, train_mask):
        # shapes: q1,q2 (N,T,1 or N,T,*), r,v_next,dones,visible,next_visible (N,T,1)
        N, T = visible.shape[:2]
        dtype = r.dtype

        # --- Apply the mask to sanitize inputs ---
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

        # Create dense i and j tensors (as floats for pow)
        i_dense = self._buffered_sequence_arange.to(dtype)  # (N, T)
        j_dense = target_idx.to(dtype)  # (N, T)

        # Clamp j to avoid OOB errors.
        # The 'valid' mask will filter out these junk values anyway.
        j_clamped = target_idx.clamp(max=T - 1)  # (N, T)

        # n-step reward sum over [i..j] with constant gamma
        r_flat = r.squeeze(-1)  # (N,T)
        pow_t = self._buffered_pow_t  # (T,)
        rg = r_flat * pow_t  # r_t * γ^t
        S = torch.cumsum(rg, dim=1)  # prefix sum of r_t * γ^t

        # Dense R_i_j
        Sj_dense = torch.gather(S, 1, j_clamped)  # (N, T)
        S_padded = torch.nn.functional.pad(S, (1, 0), 'constant', 0.0)  # (N, T+1)
        Si_1_dense = S_padded[:, :-1]  # (N, T)
        gamma_neg_i = torch.pow(self._gamma, -i_dense)  # (N, T)
        R_i_j_dense = gamma_neg_i * (Sj_dense - Si_1_dense)  # (N, T)

        # Dense bootstrap
        vnext_j_dense = torch.gather(
            v_next, 1, j_clamped.unsqueeze(-1)
        ).squeeze(-1)  # (N, T)
        done_j_dense = torch.gather(dn, 1, j_clamped).to(dtype)  # (N, T)
        expo_dense = torch.pow(self._gamma, j_dense - i_dense + 1.0)  # (N, T)

        # Final dense target
        bootstrap_dense = expo_dense * (1.0 - done_j_dense) * vnext_j_dense  # (N, T)
        q_target_dense = (R_i_j_dense + bootstrap_dense).unsqueeze(-1)  # (N, T, 1)

        return q_target_dense, valid.unsqueeze(-1)

    def _to_tensors(self, *arrays):
        new_tensors = []
        for arr in arrays:
            if not isinstance(arr, torch.Tensor):
                try:
                    arr = torch.from_numpy(arr)
                except:
                    arr = torch.tensor(arr)
            new_tensors.append(arr.to(self._device).to(torch.float32))
        return new_tensors

    def get_initial_states(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a zero-initialized hidden state tuple for the LSTM."""
        h_0 = torch.zeros(1, batch_size, self._recurrent_hidden_size, device=self._device)
        c_0 = torch.zeros(1, batch_size, self._recurrent_hidden_size, device=self._device)
        return h_0, c_0

    def _log_progress(self, epoch: int, evaluators: dict = None):
        assert evaluators is not None, \
            "Evaluators must be provided for logging."

        log_rewards = {}
        eval_str = '\n' + '=' * 40 + f"\nEpoch {epoch}:"
        self.eval()
        if evaluators is not None:
            for key in evaluators.keys():
                episodic_rewards = evaluators[key](self, seed=self._seed)

                episodic_rewards = np.array(episodic_rewards)
                log_rewards[key] = episodic_rewards.mean()

                # Get IQM for printout only
                iqr = trimboth(episodic_rewards, proportiontocut=0.25)
                iqr_mean = np.mean(iqr)
                iqr_std = np.std(iqr)
                iqr_n_samples = len(iqr)

                log_rewards[key + '_IQM'] = iqr_mean

                eval_str += f"\n     {key} = {iqr_mean:.2f} +/- {iqr_std / np.sqrt(iqr_n_samples):.2f}"

        eval_str += '=' * 40 + '\n'
        print(eval_str)
        self.train()

        return log_rewards

    @staticmethod
    def _reset_loss_dict():
        return defaultdict(partial(deque, maxlen=100))

    def _save_best_model_state(self):
        """Saves a deep copy of all nn.Module state_dicts in memory."""
        self._best_model_state = deepcopy(self.state_dict())

    def _load_best_model_state(self):
        """Loads the previously saved 'best' model state_dicts from memory."""
        if not hasattr(self, '_best_model_state') or self._best_model_state is None:
            # No state saved, nothing to do.
            return
        self.load_state_dict(self._best_model_state)

    def save_checkpoint(self, filepath: str):
        """
        Saves the state of all models, optimizers, and the GradScaler.
        """
        try:
            torch.save(self.state_dict(), filepath)
            print(f"✅ Checkpoint saved successfully to {filepath}")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")

    def load_checkpoint(self, filepath: str):
        """
        Loads the state of all models, optimizers, and the GradScaler.
        """
        try:
            # Load checkpoint to the model's device
            checkpoint = torch.load(filepath, map_location=self._device)
        except FileNotFoundError:
            print(f"❌ Error: No checkpoint file found at {filepath}")
            return
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return

        try:
            self.load_state_dict(checkpoint)
        except Exception as e:
            print(f"⚠️ Warning: Could not load state_dict directly: {e}")

        print(f"✅ Checkpoint loaded from {filepath}.")


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

        # Define steps_per_epoch - should map based on decoy_interval
        if self._cloning_only:
            self.steps_per_epoch = {0: 1_000, 1: 1_000, 2: 500}
        else:
            self.steps_per_epoch = {0: 1_000, 1: 1_000, 2: 500}

        # Set our update functions
        if not self._cloning_only:
            self.update_funcs.update({
                'critic_loss': self.update_critic,
                'value_loss': self.update_value
            })
        self.update_funcs.update({'policy_loss': self.update_actor})

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

    @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def _update_critic_compiled(self, acts, lstm_out_q, next_lstm_out_tgt):
        # We only need the network output, not the final hidden state for training
        q1 = self.critic_net1(lstm_out_q, actions=acts)
        q2 = self.critic_net2(lstm_out_q, actions=acts)

        with torch.no_grad():
            v_next = self.target_value_net(next_lstm_out_tgt, actions=None)

        return q1, q2, v_next

    def _update_critic(self, obs, acts, rews, dones, next_obs, next_acts,
                   visible, next_visible, padding_mask, next_padding_mask, train_mask):
        # Add some noise
        obs = self.add_noise(obs)
        next_obs = self.add_noise(next_obs)

        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            # Do our LSTM calls outside the compiled function
            lstm_out_q, hidden_state = self.shared_critic_encoder(obs, train_mask=train_mask)
            with torch.no_grad():
                next_lstm_out_tgt, _ = self.target_shared_critic_encoder(next_obs, train_mask=None)

            q1, q2, v_next = self._update_critic_compiled(acts, lstm_out_q, next_lstm_out_tgt)

            if self.decoy_interval == 0:
                q_target, train_mask = self._filter_to_correct_visibles(
                    rews, v_next, dones, visible, next_visible, padding_mask, train_mask)

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

    @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def _update_value_compiled(self, acts, lstm_out_q):
        v = self.value_net(lstm_out_q, actions=None)

        with torch.no_grad():
            q1 = self.critic_net1(lstm_out_q, actions=acts)
            q2 = self.critic_net2(lstm_out_q, actions=acts)
            q = torch.min(q1, q2)

        return q, v

    def _update_value(self, obs, acts, rews, dones, next_obs, next_acts,
                   visible, next_visible, padding_mask, next_padding_mask, train_mask):
        # Add some noise
        obs = self.add_noise(obs)

        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            # Do our LSTM calls outside the compiled function
            lstm_out_q, _ = self.shared_critic_encoder(obs, train_mask=train_mask)

            q, v = self._update_value_compiled(acts, lstm_out_q)

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
        policy_loss = self._update_actor(*batch, diff)
        policy_loss = self.generic_update(
            policy_loss,
            self.policy_optim,
            self.policy_net, self.actor_encoder
        )

        return policy_loss

    @torch.compile(mode='default')
    def _update_actor_compiled(self, acts, lstm_out_pi):
        params = self.policy_net(lstm_out_pi, actions=None)

        dist = self.dist.proba_distribution(params)

        # Calculate the log-probability of the expert actions `acts`
        log_pi = dist.log_prob(acts)

        # The policy loss is the Negative Log-Likelihood (NLL)
        policy_loss_unmasked = -log_pi

        return policy_loss_unmasked

    def _update_actor(self, obs, acts, rews, dones, next_obs, next_acts,
                   visible, next_visible, padding_mask, next_padding_mask, train_mask, diff):
        # Add some noise
        obs = self.add_noise(obs)

        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            # Policy doesn't depend on actions
            lstm_out_pi, _ = self.actor_encoder(obs, train_mask=train_mask)

            policy_loss_unmasked = self._update_actor_compiled(acts, lstm_out_pi)

            # Get the IQL weights
            weights = 1.0
            if not self._cloning_only:
                weights = torch.clip(torch.exp(self._beta * diff), 0, 100)

            # Apply the mask
            final_mask = train_mask & visible
            masked_loss = policy_loss_unmasked * weights * final_mask
            policy_loss = masked_loss.sum() / (final_mask.sum() + self._eps)

        return policy_loss


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
        self._entropy_alpha = torch.tensor(0.0, device=self._device)
        self._cql_alpha = torch.tensor(1.0, device=self._device)
        self.cql_uniform_log_probs = -torch.log(torch.tensor(INSULIN_ACTION_HIGH - INSULIN_ACTION_LOW,
                                                             device=self._device))

        # Define steps_per_epoch - should map based on decoy_interval
        self.steps_per_epoch = {0: 500, 1: 500, 2: 250}

        # Set our update functions
        self.update_funcs.update({'policy_loss': self.update_actor,
                                  'critic_loss': self.update_critic})

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

    @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def _update_critic_inference_compiled(self, obs, acts, visible, train_mask,
                                          lstm_out_q, lstm_out_pi, next_lstm_out_pi, next_lstm_out_tgt):
        # -- Inference for Bellman loss -- #
        # lstm_out_q, _ = self.shared_critic_encoder(obs, train_mask=train_mask)
        q1_pred = self.critic_net1(lstm_out_q, actions=acts)
        q2_pred = self.critic_net2(lstm_out_q, actions=acts)

        with torch.no_grad():
            # 2. Get our a' sampled from our policy
            params = self.policy_net(next_lstm_out_pi, actions=None)

            # Create the Beta distribution
            dist = self.dist.proba_distribution(params)

            # Sample next action
            next_actions_sampled = dist.sample()
            next_log_pi = dist.log_prob(next_actions_sampled)

            # 3. Get our Q(s',a') using the sampled action
            next_q1 = self.target_critic_net1(next_lstm_out_tgt, actions=next_actions_sampled)
            next_q2 = self.target_critic_net2(next_lstm_out_tgt, actions=next_actions_sampled)
            next_q = torch.min(next_q1, next_q2)

            # Add entropy regularisation
            next_v = next_q - self._entropy_alpha * next_log_pi

        # -- Inference for CQL loss -- #
        batch_size, seq_len, obs_dim = obs.shape

        # Define sample counts (Split 3 ways)
        n_curr = 7
        n_next = 7
        n_unif = 6

        with torch.no_grad():
            # A. Uniform Samples
            cql_unif_actions = torch.empty(n_unif, batch_size, seq_len, 1, device=obs.device)
            cql_unif_actions.uniform_(INSULIN_ACTION_LOW, INSULIN_ACTION_HIGH)
            cql_unif_log_probs = torch.log(
                torch.tensor(1.0 / (INSULIN_ACTION_HIGH - INSULIN_ACTION_LOW), device=obs.device))
            cql_unif_log_probs = cql_unif_log_probs.expand(n_unif, batch_size, seq_len, 1)

            # B. Current Policy Samples (using cql_lstm_out_pi)
            params_curr = self.policy_net(lstm_out_pi, actions=None)
            dist_curr = self.dist.proba_distribution(params_curr)
            cql_curr_actions = dist_curr.sample((n_curr,))
            cql_curr_log_probs = dist_curr.log_prob(cql_curr_actions)

            # C. Next Policy Samples (using next_lstm_out_pi)
            # Note: We reuse the next_lstm_out_pi passed in for the Bellman step
            params_next = self.policy_net(next_lstm_out_pi, actions=None)
            dist_next = self.dist.proba_distribution(params_next)
            cql_next_actions = dist_next.sample((n_next,))
            cql_next_log_probs = dist_next.log_prob(cql_next_actions)

            # Concatenate all sampled actions
            cql_sampled_actions = torch.cat([cql_curr_actions, cql_next_actions, cql_unif_actions], 0)
            cql_sampled_log_probs = torch.cat([cql_curr_log_probs, cql_next_log_probs, cql_unif_log_probs], 0)

        # Calculate the Q values for sampled actions
        total_samples = n_curr + n_next + n_unif

        # Expand hidden state to match sample dimension
        lstm_out_q_expanded = lstm_out_q.unsqueeze(0).expand(total_samples, -1, -1, -1)

        # Flatten for batch processing
        flat_hidden = lstm_out_q_expanded.reshape(-1, seq_len, lstm_out_q.shape[-1])
        flat_actions = cql_sampled_actions.reshape(-1, seq_len, cql_sampled_actions.shape[-1])

        flat_q1 = self.critic_net1(flat_hidden, actions=flat_actions)
        flat_q2 = self.critic_net2(flat_hidden, actions=flat_actions)

        # Reshape back to [n_samples, batch, seq, 1]
        cql_q1_sampled = flat_q1.view(total_samples, batch_size, seq_len, 1)
        cql_q2_sampled = flat_q2.view(total_samples, batch_size, seq_len, 1)

        # Importance sampling correct (Q - log_pi)
        cql_q1_sampled = cql_q1_sampled - cql_sampled_log_probs
        cql_q2_sampled = cql_q2_sampled - cql_sampled_log_probs

        # Add in the current data Q values
        cql_q1_combined = torch.cat([cql_q1_sampled, q1_pred.unsqueeze(0)], dim=0)
        cql_q2_combined = torch.cat([cql_q2_sampled, q2_pred.unsqueeze(0)], dim=0)

        # Calculate the logsumexp
        temperature = 1.0
        cql_q1_combined = cql_q1_combined / temperature
        cql_q2_combined = cql_q2_combined / temperature

        cql_q1_logsumexp = torch.logsumexp(cql_q1_combined / temperature, dim=0) * temperature
        cql_q2_logsumexp = torch.logsumexp(cql_q2_combined / temperature, dim=0) * temperature

        # Log(N) normalization:
        cql_q1_logsumexp = cql_q1_logsumexp - np.log(total_samples + 1)
        cql_q2_logsumexp = cql_q2_logsumexp - np.log(total_samples + 1)

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

    def _update_critic(self, obs, acts, rews, dones, next_obs, next_acts,
                   visible, next_visible, padding_mask, next_padding_mask, train_mask):
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
        # Add some noise
        obs = self.add_noise(obs)
        next_obs = self.add_noise(next_obs)

        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):

            # Do our LSTM inferences (already optimised under cudnn, slows down .compile)
            lstm_out_q, _ = self.shared_critic_encoder(obs, train_mask=train_mask)
            with torch.no_grad():
                lstm_out_pi, _ = self.actor_encoder(obs, train_mask=None)
                next_lstm_out_pi, _ = self.actor_encoder(next_obs, train_mask=None)
                next_lstm_out_tgt, _ = self.target_shared_critic_encoder(next_obs, train_mask=None)

            q1_pred, q2_pred, next_v, cql_loss = self._update_critic_inference_compiled(
                obs, acts, visible, train_mask,
                lstm_out_q, lstm_out_pi, next_lstm_out_pi, next_lstm_out_tgt
            )

            # Create the Bellman loss
            if self.decoy_interval == 0:
                q_target, train_mask = self._filter_to_correct_visibles(
                    rews, next_v, dones, visible, next_visible, padding_mask, train_mask)

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

    @torch.compile(mode='default')  # Cannot use cudagraphs here AND at critic - so we choose critic (slower)
    def _update_actor_compiled(self, acts, lstm_out_pi, lstm_out_q):
        params = self.policy_net(lstm_out_pi, actions=None)

        # Create the Beta distribution
        dist = self.dist.proba_distribution(params)

        # Sample an action (rsample) in [low, high] range
        actions_sampled, log_pi_sampled = dist.sample_and_logprob()

        # Get Q-values for these new actions
        # We pass u_sampled (normalized action) to the critics, as they expect.
        # We do *not* detach gradients here, as we need them to flow through Q into u_sampled
        with torch.no_grad():
            q1_dataset = self.critic_net1(lstm_out_q, actions=acts)
            q2_dataset = self.critic_net2(lstm_out_q, actions=acts)
            q_values = torch.min(q1_dataset, q2_dataset)

        q1_sampled = self.critic_net1(lstm_out_q, actions=actions_sampled)
        q2_sampled = self.critic_net2(lstm_out_q, actions=actions_sampled)
        q_sampled = torch.min(q1_sampled, q2_sampled)

        return log_pi_sampled, q_sampled, q_values

    def _update_actor(self, obs, acts, rews, dones, next_obs, next_acts,
                   visible, next_visible, padding_mask, next_padding_mask, train_mask):
        # Add some noise
        obs = self.add_noise(obs)

        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            # 1. Get the LSTM output for the policy (faster if not compiled)
            lstm_out_pi, _ = self.actor_encoder(obs, train_mask=train_mask)
            with torch.no_grad():
                lstm_out_q, _ = self.shared_critic_encoder(obs, train_mask=None)

            log_pi_sampled, q_sampled, q_values = self._update_actor_compiled(
                acts, lstm_out_pi, lstm_out_q
            )

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


class RecurrentFQE(_RecurrentBase):
    def __init__(
            self,
            target_model: Union[RecurrentIQL, RecurrentCQLSAC],
            tau_target: float = 0.005,
            cql_alpha: float = 0.01,
            *args,
            **kwargs
    ):
        # We set gamma = 1.0 because we want the undiscounted estimated returns.
        kwargs.pop('gamma', None)
        super().__init__(gamma=1.0, *args, **kwargs)
        self._tau_target = tau_target
        self._cql_alpha = torch.tensor(cql_alpha, device=self._device)

        # Define steps_per_epoch - should map based on decoy_interval
        self.steps_per_epoch = {0: 5_000, 1: 5_000, 2: 5_000}

        # Set our update functions
        self.update_funcs.update({'critic_loss': self.update_critic})

        # 1. Handle the Target Model (The policy we are evaluating)
        self.target_model = target_model
        self.target_model.to(self._device)
        self.target_model.eval()
        # Freeze the target model parameters completely
        for param in self.target_model.parameters():
            param.requires_grad = False

        # 2. Setup FQE Network Architecture
        # We need a specific encoder/decoder for the Q-function we are learning
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

        # --- FQE Encoders (Current and Target) ---
        self.fqe_encoder = SharedRecurrentEncoder(
            feature_extractor=FeatureEncoder(**encoder_kwargs).to(self._device),
            **encoder_kwargs
        ).to(self._device)

        self.fqe_target_encoder = SharedRecurrentEncoder(
            feature_extractor=FeatureEncoder(**encoder_kwargs).to(self._device),
            **encoder_kwargs
        ).to(self._device)

        # --- FQE Decoders (Current and Target) ---
        # We use 2 critics by default to reduce overestimation bias, standard in FQE
        self.q_net = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)
        self.q_target_net = RecurrentNet(output_size=1, has_action_encoder=True, **decoder_kwargs).to(self._device)

        # --- Optimizers ---
        # We only optimize the FQE Q-networks
        self.critic_optim = torch.optim.Adam(
            list(self.fqe_encoder.parameters())
            + list(self.q_net.parameters()),
            lr=self._critic_lr
        )

        # Initialize target networks
        self.sync_target_networks(tau=1.0)

    def update_critic(self, *batch):
        # Standard update wrapper
        critic_loss = self._update_critic(*batch)

        # Update parameters
        critic_loss = self.generic_update(
            critic_loss,
            self.critic_optim,
            self.q_net, self.fqe_encoder
        )
        return critic_loss

    @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def _update_critic_compiled(self, obs, acts, next_acts, visible, train_mask,
                                lstm_out_q, lstm_out_pi, next_lstm_out_pi, next_lstm_out_tgt):
        # 1. Compute Current Q(s, a)
        q_preds = self.q_net(lstm_out_q, actions=acts)

        # 2. Compute Target Q(s', pi(s'))
        with torch.no_grad():
            # Get the next action from the target policy
            next_params = self.target_model.policy_net(next_lstm_out_pi, actions=None)

            # Create the Beta distribution
            next_dist = self.target_model.dist.proba_distribution(next_params)

            # Get the deterministic next action
            next_action = next_dist.mode()

            if next_action is None:
                next_action = next_acts

            # Compute Q_target values using target Q nets
            next_q_preds = self.q_target_net(next_lstm_out_tgt, actions=next_action)

        # CQL OOD Penalty calcuation
        # We want to penalise LogSumExp(Q(s, a_sampled)) - Q(s, a_data)
        batch_size, seq_len, _ = obs.shape
        n_unif = 5
        n_policy = 5
        total_samples = n_unif + n_policy

        with torch.no_grad():
            # Uniform Samples (Global OOD)
            cql_unif_actions = torch.empty(n_unif, batch_size, seq_len, 1, device=obs.device)
            cql_unif_actions.uniform_(INSULIN_ACTION_LOW, INSULIN_ACTION_HIGH)

            # Target Policy Samples
            # We use the current state policy hidden state passed in (lstm_out_pi_target)
            params = self.target_model.policy_net(lstm_out_pi, actions=None)
            dist = self.target_model.dist.proba_distribution(params)
            cql_policy_actions = dist.sample((n_policy,))

            # Concatenate
            cql_sampled_actions = torch.cat([cql_policy_actions, cql_unif_actions], 0)

        # Expand hidden state to match sample dimension
        lstm_out_q_expanded = lstm_out_q.unsqueeze(0).expand(total_samples, -1, -1, -1)

        # Flatten for batch processing
        flat_hidden = lstm_out_q_expanded.reshape(-1, seq_len, lstm_out_q.shape[-1])
        flat_actions = cql_sampled_actions.reshape(-1, seq_len, cql_sampled_actions.shape[-1])

        # Get Q-values for sampled actions
        flat_q_ood = self.q_net(flat_hidden, actions=flat_actions)

        # Reshape back to [n_samples, batch, seq, 1]
        cql_q_sampled = flat_q_ood.view(total_samples, batch_size, seq_len, 1)

        # Add in current data for the LogSumExp
        cql_q_combined = torch.cat([cql_q_sampled, q_preds.unsqueeze(0)], dim=0)

        # Calculate the logsumexp
        temperature = 1.0
        cql_q_combined = cql_q_combined / temperature
        cql_q_logsumexp = torch.logsumexp(cql_q_combined / temperature, dim=0) * temperature

        # Log(N) normalization
        cql_q_logsumexp = cql_q_logsumexp - np.log(total_samples + 1)

        # CQL Loss
        cql_loss = self._cql_alpha * (cql_q_logsumexp - q_preds)

        # Use train_mask (which is False for padding AND burn-in)
        final_mask = train_mask * visible  # Shape (N, T)
        valid_count = final_mask.sum() + self._eps

        scaled_cql_loss = cql_loss * final_mask
        scaled_cql_loss = scaled_cql_loss.sum() / valid_count

        return q_preds, next_q_preds, scaled_cql_loss

    def _update_critic(self, obs, acts, rews, dones, next_obs, next_acts,
                   visible, next_visible, padding_mask, next_padding_mask, train_mask):
        # Add noise (consistency with IQL/CQL classes)
        obs = self.add_noise(obs)
        next_obs = self.add_noise(next_obs)

        with torch.autocast(device_type="cuda", enabled=self.scaler is not None, dtype=self._scaler_dtype):
            # Do LSTM inference outside of compiled loop
            lstm_out_q, _ = self.fqe_encoder(obs, train_mask=train_mask)
            with torch.no_grad():
                # Get the next action from the target policy
                lstm_out_pi, _ = self.target_model.actor_encoder(obs, train_mask=None)
                next_lstm_out_pi, _ = self.target_model.actor_encoder(next_obs, train_mask=None)
                next_lstm_out_tgt, _ = self.fqe_target_encoder(next_obs, train_mask=None)

            current_q, next_q, scaled_cql_loss = self._update_critic_compiled(
                obs, acts, next_acts, visible, train_mask,
                lstm_out_q, lstm_out_pi, next_lstm_out_pi, next_lstm_out_tgt
            )

            # Calculate Bellman Target
            # y = r + gamma * (1-d) * Q_target
            if self.decoy_interval == 0:
                # If using specialized filtering from base class
                q_target, train_mask = self._filter_to_correct_visibles(
                    rews, next_q, dones, visible, next_visible, padding_mask, train_mask)
            else:
                q_target = rews + self._gamma * (1 - dones.float()) * next_q

            # MSE Loss
            critic_loss_unmasked = F.mse_loss(current_q, q_target, reduction='none')

            # Apply Mask
            # Use train_mask (which is False for padding AND burn-in)
            masked_loss = critic_loss_unmasked * train_mask
            critic_loss = masked_loss.sum() / (train_mask.sum() + self._eps)

        return critic_loss + scaled_cql_loss

    def sync_target_networks(self, tau=None):
        """Soft updates for the FQE specific networks"""
        tau = tau or self._tau_target

        # Sync Encoder
        for target_param, param in zip(self.fqe_target_encoder.parameters(), self.fqe_encoder.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

        # Sync Decoders
        for target_param, param in zip(self.q_target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    def predict(self, obs: np.ndarray, hidden_state=None, deterministic: bool = False, with_dist: bool = False,
                action_as_tensor: bool = False, *args, **kwargs):
        """
        Delegates prediction to the target model.
        When we evaluate RecurrentFQE, we are evaluating the behavior of the target_model.
        """
        raise NotImplementedError("Not implemented for FQE")

    def get_value_estimate(self, obs, acts):
        """
        Helper to get the FQE estimated Q-value for a specific batch.
        Useful for debugging or plotting OPE results.
        """
        obs_tensor = self._to_tensors(obs)[0] if not isinstance(obs, torch.Tensor) else obs
        acts_tensor = self._to_tensors(acts)[0] if not isinstance(acts, torch.Tensor) else acts

        with torch.no_grad():
            lstm_out, _ = self.fqe_encoder(obs_tensor, train_mask=None)
            q_preds = self.q_net(lstm_out, actions=acts_tensor)
        return q_preds
