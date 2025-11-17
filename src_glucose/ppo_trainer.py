from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import IterableDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import gymnasium as gym
import numpy as np
import os
from gymnasium.vector import AsyncVectorEnv
from torch.distributions import Beta
from functools import partial
import random
from tqdm import tqdm
from gym_wrappers import INSULIN_ACTION_LOW, INSULIN_ACTION_HIGH, MASTER_SEED, TOTAL_SIZE

LSTM_LAYERS = 2


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across torch, numpy, and random.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for PyTorch on CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Set deterministic algorithms for PyTorch (can impact performance)
    # This is crucial for full reproducibility with CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # You might also want to set this for newer PyTorch versions
    # torch.use_deterministic_algorithms(True)

    # Set environment variable for CUDA (if needed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


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
            layer_norm_out: bool = True,
            out_dim: Optional[int] = None,
    ):
        super().__init__()

        out_dim = out_dim or hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        if layer_norm_out:
            self.fc.append(nn.LayerNorm(out_dim))
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

    def proba_distribution(self, params: torch.Tensor):
        """
        Creates the distribution from the actor network's output.

        :param params: Raw output from the actor head (batch_size, action_dim * 2)
        """
        # Split the output into alpha and beta parameters
        alpha, beta = torch.chunk(params, 2, dim=-1)

        # Softplus to keep alpha, beta > 1 for unimodal distribution
        self.alpha = F.softplus(alpha) + 1.0
        self.beta = F.softplus(beta) + 1.0

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
        return log_prob


class EncoderActorCriticLSTM(nn.Module):
    """
    An Actor-Critic network that first encodes observations using a feature extractor
    and then processes the sequence of encoded features with an LSTM.
    """

    def __init__(self, encoder: nn.Module, encoder_output_dim: int, hidden_dim: int, action_dim: int):
        super(EncoderActorCriticLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # The feature extractor
        self.encoder = encoder

        # The LSTM now takes the output of the encoder as its input
        self.lstm = nn.LSTM(encoder_output_dim, hidden_dim, batch_first=True, num_layers=LSTM_LAYERS)

        self.actor_head = FeatureEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=action_dim * 2,
            layer_norm_out=False
        )
        self.critic_head = FeatureEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            layer_norm_out=False
        )

        self.dist = CustomBetaDistribution(
            action_dim,
            low=INSULIN_ACTION_LOW,
            high=INSULIN_ACTION_HIGH,
        )

    def forward_lstm(self, x, hidden_state=None, unsorted_indices=None, padding_mask=None):
        # Need to use packed sequence for PPO because c_n must be accurate (as it gets re-used with each new step)
        # - doesn't apply for offline learning because we never need to re-use hidden states.
        is_packed = isinstance(x, PackedSequence)
        assert not (is_packed and padding_mask is not None), "Padding mask has no effect on packed tensors."

        if is_packed:
            # If the input is a packed sequence, apply the encoder to the data part
            encoded_data = self.encoder(x.data)
            # Create a new packed sequence with the encoded data
            lstm_input = PackedSequence(encoded_data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
        else:
            # Handle non-packed sequences (e.g., during prediction/evaluation)
            batch_size, seq_len, feature_dim = x.shape
            # Reshape for the encoder: (batch * seq_len, features)
            x_reshaped = x.reshape(batch_size * seq_len, feature_dim)
            # Encode the features
            encoded_x = self.encoder(x_reshaped)
            # Reshape back for the LSTM: (batch, seq_len, encoder_output_dim)
            lstm_input = encoded_x.reshape(batch_size, seq_len, -1)

        if padding_mask is not None and not is_packed:
            # Detach using padding mask to reduce gradient calculation requirements
            lstm_input = torch.where(padding_mask, lstm_input, lstm_input.detach())

        lstm_out, new_hidden = self.lstm(lstm_input, hidden_state)

        if is_packed:
            sorted_lstm_out, sorted_lengths = pad_packed_sequence(lstm_out, batch_first=True)

            # Unsort to match the original input order
            unsorted_indices = x.unsorted_indices if x.unsorted_indices is not None else unsorted_indices
            lstm_out = sorted_lstm_out.index_select(dim=0, index=unsorted_indices)
            lengths = sorted_lengths.index_select(dim=0, index=unsorted_indices)

            new_hidden = (new_hidden[0].index_select(dim=1, index=unsorted_indices),
                          new_hidden[1].index_select(dim=1, index=unsorted_indices))

            # Get the last outputs
            batch_size = lstm_out.size(0)
            last_step_indices = lengths.long() - 1
            last_outputs = lstm_out[torch.arange(batch_size), last_step_indices, :]

        else:
            last_outputs = lstm_out[:, -1, :]

        return lstm_out, last_outputs, new_hidden

    def forward(self, x, hidden_state=None, deterministic=False, unsorted_indices=None):
        lstm_out, last_outputs, new_hidden = self.forward_lstm(
            x, hidden_state=hidden_state, unsorted_indices=unsorted_indices
        )

        value = self.critic_head(last_outputs)
        actor_out = self.actor_head(last_outputs)

        dist = self.dist.proba_distribution(actor_out)

        if deterministic:
            action = dist.mode()
            return action, value, new_hidden

        sampled_action = dist.sample()
        # Store the action so the training loop can get it
        self.dist.last_sampled_action = sampled_action

        return dist, value, new_hidden

    def init_hidden_state(self, batch_size=1):
        return torch.zeros(LSTM_LAYERS, batch_size, self.hidden_dim), torch.zeros(LSTM_LAYERS, batch_size,
                                                                                  self.hidden_dim)


class RecurrentPPO:
    def __init__(
            self,
            # Env params
            train_env_creator_fn,
            eval_env_creator_fn,
            train_ids, test_ids,
            train_envs_per_id: int = 1,
            eval_envs_per_id: int = 1,
            # Network params
            hidden_dim=128,
            learning_rate=3e-4,
            device='cpu',
            # Learning params
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.01,
            vf_coef=0.5,
            clip_range=0.2,
            seed=None,
            # Sampling params
            n_steps=2048,
            n_epochs=10,
            n_minibatches=8,
            batch_sequence_length=64,
            # Eval params
            eval_freq=10000,
            eval_episodes=500,
            log_dir="../logs_glucose/ppo_logs"):
        # Store device, but keep everything on CPU initially
        self.device = device
        # Set the seed
        self.seed = seed
        if seed is not None:
            self.set_random_seed(seed)

        # Save the patient IDs and env_creator functions
        self.train_ids = train_ids
        self.test_ids = test_ids

        self.train_envs_per_id = train_envs_per_id
        self.eval_envs_per_id = eval_envs_per_id
        self.n_train_envs = len(self.train_ids) * train_envs_per_id
        self.n_eval_envs = len(self.test_ids) * eval_envs_per_id

        self.train_env_creator_fn = partial(train_env_creator_fn, n_envs=self.n_train_envs, gamma=gamma)
        self.eval_env_creator_fn = partial(eval_env_creator_fn, gamma=gamma)

        dummy_env = train_env_creator_fn(patient_ids=self.train_ids)

        # Save our env parameters
        assert isinstance(dummy_env.action_space, gym.spaces.Box), "Continuous action spaces only."

        self.input_dim = dummy_env.observation_space.shape[-1]
        self.action_dim = dummy_env.action_space.shape[0]
        self.action_scale = torch.FloatTensor((dummy_env.action_space.high - dummy_env.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((dummy_env.action_space.high + dummy_env.action_space.low) / 2.)
        self.hidden_dim = hidden_dim

        # Create our vectorised environments
        self.train_env, self.train_env_id_map = self._build_parallel_env(self.train_env_creator_fn,
                                                                         self.train_ids,
                                                                         self.train_envs_per_id)
        self.eval_env, self.eval_env_id_map = self._build_parallel_env(self.eval_env_creator_fn,
                                                                       self.test_ids,
                                                                       self.eval_envs_per_id)

        # Save the rest of our training parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.clip_range = clip_range

        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches  # Number of sequences per minibatch
        self.batch_sequence_length = batch_sequence_length  # Num of decisions per sequence

        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.log_dir = log_dir
        self.last_obs_tensor = None
        self.last_hidden_state = None
        self.last_unsorted_idx = None

        self._num_timesteps = 0
        self.best_mean_reward = -np.inf
        self.mean_ep_length = 0.0

        # Create log directory if it doesn't exist
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        self.encoder = FeatureEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        self.ac_network = EncoderActorCriticLSTM(
            encoder=self.encoder,
            encoder_output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
        )
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=self.learning_rate)

    def __call__(self, obs, deterministic=True, *args, **kwargs):
        action, lstm_states = self.predict(obs, deterministic=deterministic, *args, **kwargs)
        return action, lstm_states

    @staticmethod
    def _build_parallel_env(env_creator_fn, patient_ids, envs_per_id):
        # Create the vectorized environment
        parallel_env = AsyncVectorEnv([
            partial(env_creator_fn, patient_ids=pid)
            for pid in patient_ids
            for _ in range(envs_per_id)
        ])
        # Create the mapping: [0] -> 'id_A', [1] -> 'id_A', ..., [7] -> 'id_A', [8] -> 'id_B', ...
        id_map = [
            pid for pid in patient_ids
            for _ in range(envs_per_id)
        ]
        return parallel_env, id_map

    def get_initial_states(self, batch_size: int):
        """Returns a zero-initialized hidden state tuple for the LSTM."""
        return self.ac_network.init_hidden_state(batch_size)

    def set_random_seed(self, seed):
        """Sets the random seed for reproducibility."""
        self.seed = seed
        set_seed(seed)

    def save_checkpoint(self, path):
        """Saves the model and optimizer state."""
        checkpoint = {
            'ac_network_state_dict': self.ac_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_timesteps': self._num_timesteps,
            'best_mean_reward': self.best_mean_reward,
            'mean_ep_length': self.mean_ep_length,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path, env, *args, **kwargs):
        """Loads the model and optimizer state."""
        assert os.path.exists(path), f"No checkpoint found at {path}"

        # Get the class instance
        new_agent = cls(env, *args, **kwargs)

        # Update the state
        checkpoint = torch.load(path, weights_only=False)
        new_agent.ac_network.load_state_dict(checkpoint['ac_network_state_dict'])
        new_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        new_agent._num_timesteps = checkpoint.get('num_timesteps', 0)
        new_agent.best_mean_reward = checkpoint.get('best_mean_reward', -np.inf)
        new_agent.mean_ep_length = checkpoint.get('mean_ep_length', 0.0)
        print(
            f"Checkpoint loaded from {path}. Resuming at timestep {new_agent._num_timesteps} with best IQM reward {new_agent.best_mean_reward:.2f} and mean length {new_agent.mean_ep_length:.2f}.")

        return new_agent

    def predict(self, obs, hidden_state=None, deterministic=True):
        """
        Get the model's action for a given observation.

        :param obs: The current observation (should be a sequence)
        :param hidden_state: The last hidden state of the LSTM
        :param deterministic: Whether to return a deterministic or stochastic action
        :return: the model's action and the next hidden state
        """
        # obs is expected to be of shape (seq_len, features)
        # Add a batch dimension -> (1, seq_len, features)
        obs_tensor = torch.FloatTensor(obs)
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                # Use the mode for deterministic prediction
                action, _, new_hidden_state = self.ac_network(
                    obs_tensor,
                    hidden_state=hidden_state,
                    deterministic=True
                )
            else:
                # Sample for stochastic action
                dist, _, new_hidden_state = self.ac_network(
                    obs_tensor,
                    hidden_state=hidden_state,
                    deterministic=False
                )
                action = dist.sample()

        # The environment expects a flattened numpy array
        action = action.cpu().numpy().flatten()
        return action, new_hidden_state

    def _compute_advantages_and_returns(self, rewards, values, dones, steps_taken):
        n_steps = len(rewards)
        advantages = np.zeros((n_steps, self.n_train_envs), dtype=np.float32)
        last_advantage = np.zeros(self.n_train_envs, dtype=np.float32)

        with torch.no_grad():
            _, last_value, _ = self.ac_network(
                self.last_obs_tensor,
                hidden_state=self.last_hidden_state,
                unsorted_indices=self.last_unsorted_idx
            )

        last_value = last_value.squeeze().numpy()

        for t in reversed(range(n_steps)):
            mask = 1.0 - dones[t]
            effective_gamma = self.gamma ** steps_taken[t]
            delta = rewards[t] + effective_gamma * last_value * mask - values[t]
            last_advantage = delta + effective_gamma * self.gae_lambda * last_advantage * mask
            advantages[t] = last_advantage
            last_value = values[t]

        returns = advantages + np.stack(values)
        return advantages, returns

    def _evaluate_policy(self, n_eval_episodes=None, env_creator_fn=None, test_ids=None):
        # (optional) Rebuild eval envs if a custom env_creator_fn or test_ids are provided
        if env_creator_fn is not None or test_ids is not None:
            if env_creator_fn is not None:
                print('Overriding default env_creator_fn with provided env_creator_fn.')

            eval_env_creator_fn = env_creator_fn or self.eval_env_creator_fn

            if test_ids is not None:
                print('Overriding default test_ids with provided test_ids.')

            test_ids = test_ids or self.test_ids
            n_eval_envs = len(test_ids) * self.eval_envs_per_id

            eval_env, eval_env_id_map = self._build_parallel_env(eval_env_creator_fn,
                                                                 test_ids,
                                                                 self.eval_envs_per_id)
        else:
            test_ids = self.test_ids
            eval_env = self.eval_env
            eval_env_id_map = self.eval_env_id_map
            n_eval_envs = self.n_eval_envs

        n_eval_episodes = n_eval_episodes or self.eval_episodes

        # --- Branch for BALANCED evaluation ---
        n_ids = len(test_ids)
        # Calculate how many episodes to get from each ID
        # We use max(1, ...) to ensure we get at least one episode even if n_eval_episodes < n_ids
        target_episodes_per_id = max(1, n_eval_episodes // n_ids)
        total_target_episodes = target_episodes_per_id * n_ids

        print(f"\n--- Starting balanced evaluation for {n_ids} test IDs ---")
        print(f"--- Collecting {target_episodes_per_id} episodes per ID (Total: {total_target_episodes}) ---")

        # Dictionaries to store results per ID
        rewards_per_id = {pid: [] for pid in test_ids}
        steps_per_id = {pid: [] for pid in test_ids}

        # Trackers for in-progress episodes
        episode_rewards = np.zeros(n_eval_envs)
        episode_steps = np.zeros(n_eval_envs)

        # Seed each parallel environment with a unique, deterministic seed
        if self.seed is not None:
            rng = np.random.default_rng(MASTER_SEED)
            eval_seeds = rng.integers(low=0, high=2 ** 32 - 1, size=n_eval_envs).tolist()
            obs, _ = eval_env.reset(seed=eval_seeds)
        else:
            obs, _ = eval_env.reset()

        hidden_state = self.ac_network.init_hidden_state(batch_size=n_eval_envs)

        # --- Define Loop Condition ---
        def is_evaluation_done():
            # Loop until ALL test IDs have reached their target
            return all(len(rewards_per_id[pid]) >= target_episodes_per_id for pid in test_ids)

        # --- Main Evaluation Loop ---
        while not is_evaluation_done():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(1)
            with torch.no_grad():
                # Use stochastic actions for evaluation
                dist, _, hidden_state = self.ac_network(
                    obs_tensor,
                    hidden_state=hidden_state,
                    deterministic=False
                )
                action_env = dist.last_sampled_action

            obs, reward, term, trunc, _ = eval_env.step(action_env.numpy())
            dones = term | trunc
            episode_rewards += reward
            episode_steps += 1

            for i, done in enumerate(dones):
                if done:
                    # --- Balanced Logic: Add to the correct dictionary list ---
                    current_id = eval_env_id_map[i]

                    # Only add if this ID still needs episodes
                    if len(rewards_per_id[current_id]) < target_episodes_per_id:
                        rewards_per_id[current_id].append(episode_rewards[i])
                        steps_per_id[current_id].append(episode_steps[i])

                    # Reset this env's trackers
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                    hidden_state[0][:, i, :] = 0
                    hidden_state[1][:, i, :] = 0

        # --- Post-Loop: Combine results if we were in balanced mode ---
        all_episode_rewards = []
        all_episode_steps = []
        for pid in test_ids:
            # This guarantees an equal number of episodes from each ID
            all_episode_rewards.extend(rewards_per_id[pid])
            all_episode_steps.extend(steps_per_id[pid])
        print(f"--- Evaluation finished. Collected {len(all_episode_rewards)} total episodes. ---")

        # Calculate the IQM return
        ep_reward_array = np.array(all_episode_rewards[:total_target_episodes])
        steps_array = np.array(all_episode_steps[:total_target_episodes])

        q1_r, q3_r = np.percentile(ep_reward_array, [25, 75])
        iqr_mask = (ep_reward_array >= q1_r) & (ep_reward_array <= q3_r)

        # Handle edge case where all rewards are identical (mask is all False)
        if not iqr_mask.any():
            IQM_return = ep_reward_array.mean()
            IQM_steps = steps_array.mean()
        else:
            IQM_return = ep_reward_array[iqr_mask].mean()
            IQM_steps = steps_array[iqr_mask].mean()

        return IQM_return, {'ep_length': IQM_steps}

    def fit(self, total_timesteps):
        # Set up our initial obs
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            train_seeds = rng.integers(low=0, high=2 ** 32 - 1, size=self.n_train_envs).tolist()
            obs, info = self.train_env.reset(seed=train_seeds)
        else:
            obs, info = self.train_env.reset()

        # Pack the padded initial obs
        obs_tensor = torch.from_numpy(obs)
        packed_obs_tensor, sorted_idx, unsorted_idx = pack_obs(obs_tensor, info['steps_taken'])

        # Get our initial hidden state and sort (for best practice)
        hidden_state = (h_x, c_x) = self.get_initial_states(batch_size=self.n_train_envs)
        sorted_hidden = (h_x.index_select(dim=1, index=sorted_idx),
                         c_x.index_select(dim=1, index=sorted_idx))

        # Define our training/eval epoch parameters
        start_timesteps = self._num_timesteps
        target_timesteps = start_timesteps + total_timesteps

        timesteps_per_update = self.n_steps * self.n_train_envs
        total_updates = int(np.ceil(total_timesteps / timesteps_per_update))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_updates, eta_min=0)

        updates = 0
        model_save_num = 1
        next_eval = (start_timesteps // self.eval_freq + 1) * self.eval_freq

        while self._num_timesteps < target_timesteps:
            updates += 1

            # --- ROLLOUT COLLECTION ---
            rollout_obs, rollout_rewards, rollout_dones = [], [], []
            rollout_actions, rollout_log_probs, rollout_values = [], [], []
            rollout_h_states, rollout_c_states = [], []
            rollout_lengths = []

            current_episode_reward = np.zeros(self.n_train_envs)
            current_episode_length = np.zeros(self.n_train_envs)
            ep_rewards_this_rollout = []
            ep_lengths_this_rollout = []

            for _ in tqdm(range(self.n_steps), mininterval=2, leave=False, desc="Collecting Rollout"):
                # Store the hidden state *before* the step
                rollout_h_states.append(h_x.detach().numpy())
                rollout_c_states.append(c_x.detach().numpy())

                with torch.no_grad():
                    dist, value, hidden_state = self.ac_network(
                        packed_obs_tensor,
                        hidden_state=sorted_hidden,
                        unsorted_indices=unsorted_idx
                    )

                sampled_actions = dist.last_sampled_action
                sampled_actions_np = sampled_actions.numpy()
                log_prob = dist.log_prob(sampled_actions)

                new_obs, reward, term, trunc, info = self.train_env.step(sampled_actions_np)
                done = term | trunc
                steps_taken = info.get('steps_taken', 1)

                current_episode_reward += reward
                current_episode_length += steps_taken

                # Store rollout data
                rollout_obs.append(obs)  # obs_tensor)
                rollout_actions.append(sampled_actions_np)
                rollout_rewards.append(reward)
                rollout_dones.append(done)
                rollout_log_probs.append(log_prob.squeeze().numpy())
                rollout_values.append(value.squeeze().numpy())
                rollout_lengths.append(steps_taken)

                obs = new_obs
                if done.any():
                    ep_rewards_this_rollout += current_episode_reward[done].tolist()
                    ep_lengths_this_rollout += current_episode_length[done].tolist()
                    current_episode_reward[done] *= 0
                    current_episode_length[done] *= 0
                    hidden_state[0][:, done] *= 0
                    hidden_state[1][:, done] *= 0

                obs_tensor = torch.from_numpy(obs)
                packed_obs_tensor, sorted_idx, unsorted_idx = pack_obs(obs_tensor, steps_taken)
                (h_x, c_x) = hidden_state
                sorted_hidden = (h_x.index_select(dim=1, index=sorted_idx),
                                 c_x.index_select(dim=1, index=sorted_idx))

            self._num_timesteps += self.n_steps * self.n_train_envs

            # --- END ROLLOUT COLLECTION ---

            self.last_obs_tensor = packed_obs_tensor
            self.last_hidden_state = sorted_hidden
            self.last_unsorted_idx = unsorted_idx

            # GAE Calculation
            rollout_advantages, rollout_returns = self._compute_advantages_and_returns(rollout_rewards, rollout_values,
                                                                                       rollout_dones, rollout_lengths)

            # Normalize Advantages
            rollout_advantages = (rollout_advantages - rollout_advantages.mean(0)) / (rollout_advantages.std(0) + 1e-8)

            # --- DATA PREPARATION ---
            # Reshape data to (rollout_seq_len, n_envs, ...)
            n_steps = self.n_steps
            n_envs = self.n_train_envs

            rollout_obs = np.concatenate(rollout_obs).reshape(n_steps, n_envs, TOTAL_SIZE, -1)
            rollout_actions = np.concatenate(rollout_actions).reshape(n_steps, n_envs, 1)
            rollout_lengths = np.concatenate(rollout_lengths).reshape(n_steps, n_envs, 1)
            rollout_log_probs = np.concatenate(rollout_log_probs).reshape(n_steps, n_envs, 1)
            rollout_advantages = rollout_advantages.reshape(n_steps, n_envs, 1)
            rollout_returns = rollout_returns.reshape(n_steps, n_envs, 1)
            rollout_dones = np.array(rollout_dones).reshape(n_steps, n_envs, 1)

            rollout_h_states = (np.concatenate(rollout_h_states, 1)
                                .reshape(LSTM_LAYERS, n_steps, n_envs, -1).transpose(1, 0, 2, 3))
            rollout_c_states = (np.concatenate(rollout_c_states, 1)
                                .reshape(LSTM_LAYERS, n_steps, n_envs, -1).transpose(1, 0, 2, 3))

            sampler = LSTMSMDPBatchSampler(
                rollout_obs,  # (n_steps, n_envs, TOTAL_SIZE, 4)
                rollout_actions,  # (n_steps, n_envs, 1)
                rollout_log_probs,  # (n_steps, n_envs, 1)
                rollout_returns,  # (n_steps, n_envs, 1)
                rollout_advantages,  # (n_steps, n_envs, 1)
                rollout_dones,  # (n_steps, n_envs, 1)
                rollout_lengths,  # (n_steps, n_envs, 1)
                rollout_h_states,  # (n_steps, LSTM_LAYERS, n_envs, HIDDEN_DIM)
                rollout_c_states,  # (n_steps, LSTM_LAYERS, n_envs, HIDDEN_DIM)
                sequence_length=self.batch_sequence_length,  # The length of subsequences to sample
                n_minibatches=self.n_minibatches,  # The total number of (env, step) transitions per mini-batch
            )

            dataloader = torch.utils.data.DataLoader(
                sampler,
                batch_size=None,  # <-- Must be None for iterable-style datasets
                num_workers=2,  # <-- KEY: Number of parallel processes (start with 2-4)
                prefetch_factor=2,  # <-- KEY: Preloads 2 batches per worker
                pin_memory=True  # <-- IMPORTANT: Speeds up CPU-to-GPU transfer
            )

            # Move network to the correct device
            self.ac_network.to(self.device)

            policy_losses, value_losses, entropy_losses, approx_kls, grad_norms = [], [], [], [], []

            for _ in tqdm(
                range(self.n_epochs),
                leave=False,
                desc=f"Updating agent "
                     f"({sampler.batch_size} sequences per minibatch, "
                     f"{sampler.n_minibatches} minibatches)..."
            ):
                # The sampler shuffles data automatically on each new iteration
                for i, batch in enumerate(dataloader):
                    # Unpack the batch
                    (
                        obs_batch_cpu,
                        actions_batch_cpu,
                        log_probs_batch_cpu,
                        returns_batch_cpu,
                        advantages_batch_cpu,
                        mask_batch_cpu,
                        padding_mask_batch_cpu,
                        h_starts_cpu,
                        c_starts_cpu,
                    ) = batch

                    # --- NEW: Move batch to device *inside* the training loop ---
                    # Using non_blocking=True is efficient thanks to pin_memory=True
                    obs_batch = obs_batch_cpu.to(self.device, non_blocking=True)
                    actions_batch = actions_batch_cpu.to(self.device, non_blocking=True)
                    log_probs_batch = log_probs_batch_cpu.to(self.device, non_blocking=True)
                    returns_batch = returns_batch_cpu.to(self.device, non_blocking=True)
                    advantages_batch = advantages_batch_cpu.to(self.device, non_blocking=True)
                    mask_batch = mask_batch_cpu.to(self.device, non_blocking=True)
                    padding_mask_batch = padding_mask_batch_cpu.to(self.device, non_blocking=True)
                    h_starts = h_starts_cpu.to(self.device, non_blocking=True)
                    c_starts = c_starts_cpu.to(self.device, non_blocking=True)

                    # Run inference
                    lstm_out, _, _ = self.ac_network.forward_lstm(
                        obs_batch,
                        hidden_state=(h_starts, c_starts),
                        padding_mask=padding_mask_batch
                    )

                    new_values = self.ac_network.critic_head(lstm_out)
                    actor_out = self.ac_network.actor_head(lstm_out)
                    new_dist = self.ac_network.dist.proba_distribution(actor_out)
                    new_log_probs = new_dist.log_prob(actions_batch)

                    log_ratio = new_log_probs - log_probs_batch
                    ratio = log_ratio.exp()
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_batch

                    valid_counts = torch.clip(mask_batch.sum(), min=1)

                    policy_loss = (-torch.min(surr1, surr2) * mask_batch).sum() / valid_counts
                    value_loss = ((new_values - returns_batch) ** 2 * mask_batch).sum() / valid_counts
                    entropy_loss = (-new_dist.entropy() * mask_batch).sum() / valid_counts
                    loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * entropy_loss

                    with torch.no_grad():
                        approx_kl = ((((ratio - 1) - log_ratio) * mask_batch).sum() / valid_counts).item()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    approx_kls.append(approx_kl)

                    # 6. --- BACKWARD PASS ---
                    self.optimizer.zero_grad()
                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), 3.0)
                    grad_norms.append(total_norm)
                    self.optimizer.step()

            # Move network back to the CPU
            self.ac_network.to('cpu')

            grad_norms = torch.stack(grad_norms).cpu()

            if len(grad_norms) > 0:
                p90_grad_norm = torch.quantile(grad_norms, 0.9).item()
                mean_grad_norm = grad_norms.mean().item()  # np.mean(grad_norms)
            else:
                p90_grad_norm = 0.0
                mean_grad_norm = 0.0

            mean_reward = np.mean(ep_rewards_this_rollout) if ep_rewards_this_rollout else np.nan
            mean_ep_length = np.mean(ep_lengths_this_rollout) if ep_lengths_this_rollout else np.nan

            print(
                f"Timestep: {self._num_timesteps}/{target_timesteps} | "
                f"Mean Ep. Length {mean_ep_length:.2f} | "
                f"Mean Reward: {mean_reward:.2f} | "
                f"Grad Norm (mean): {mean_grad_norm:.3f} | "  # <--- Added here
                f"Grad Norm (P90): {p90_grad_norm:.3f} | "  # <--- Added here
                f"Entropy: {np.mean(entropy_losses):.3f} | "
                f"Value Loss: {np.mean(value_losses):.3f} | "
                f"Policy Loss: {np.mean(policy_losses):.3f} | "
                f"Approx. KL {np.mean(approx_kls):.5f}"
            )

            if self._num_timesteps >= next_eval:
                avg_eval_reward, info = self._evaluate_policy()
                print(f"--- EVALUATION at Timestep: {self._num_timesteps}/{target_timesteps} ---")
                print(
                    f"Average reward: {avg_eval_reward:.2f} | Best IQM reward: {self.best_mean_reward:.2f} | Mean Length: {self.mean_ep_length:.2f}\n")

                if avg_eval_reward > self.best_mean_reward:
                    self.best_mean_reward = avg_eval_reward
                    self.mean_ep_length = info['ep_length']
                    if self.log_dir:
                        save_path = os.path.join(self.log_dir,
                                                 f"best_model{model_save_num:02d}_{avg_eval_reward:.2f}.pth")
                        self.save_checkpoint(save_path)
                        print(f"*** New best model found and saved with reward: {self.best_mean_reward:.2f} "
                              f"and episode length {self.mean_ep_length:.2f} ***\n")
                        model_save_num += 1

                next_eval += self.eval_freq

            self.scheduler.step()

        self.train_env.close()
        self.eval_env.close()


class LSTMSMDPBatchSampler(IterableDataset):
    """
    A batch sampler for SMDP rollouts for LSTM training.

    This sampler takes (n_steps, n_envs, ...) structured rollouts and
    yields mini-batches of subsequences of shape (sequence_length, num_sequences, ...)
    along with the correct initial hidden states for each sequence.

    It correctly handles episode boundaries, ensuring no subsequence
    crosses a 'done' signal (except at the very last step).
    """

    def __init__(
            self,
            rollout_obs: np.ndarray,
            rollout_actions: np.ndarray,
            rollout_log_probs: np.ndarray,
            rollout_returns: np.ndarray,
            rollout_advantages: np.ndarray,
            rollout_dones: np.ndarray,
            rollout_lengths: np.ndarray,
            rollout_h_states: np.ndarray,
            rollout_c_states: np.ndarray,
            sequence_length: int,
            n_minibatches: int,
    ):
        """
        Initialize the sampler with the full rollout data.

        Args:
            rollout_obs: (n_steps, n_envs, TOTAL_SIZE, 4)
            rollout_actions: (n_steps, n_envs, 1)
            rollout_log_probs: (n_steps, n_envs, 1)
            rollout_returns: (n_steps, n_envs, 1)
            rollout_advantages: (n_steps, n_envs, 1)
            rollout_dones: (n_steps, n_envs, 1)
            rollout_lengths: (n_steps, n_envs, 1)
            rollout_h_states: (n_steps, LSTM_LAYERS, n_envs, HIDDEN_DIM)
            rollout_c_states: (n_steps, LSTM_LAYERS, n_envs, HIDDEN_DIM)
            sequence_length: The length of subsequences to sample (e.g., 32).
            n_minibatches: The number of minibatches
        """
        self.n_steps, self.n_envs = rollout_obs.shape[:2]
        self.sequence_length = sequence_length
        self.n_minibatches = n_minibatches

        # --- 1. Reshape data for easier sampling ---
        # We want to view the data as (n_envs, n_steps, ...)
        # This makes it easy to grab sequences from each environment.

        # (T, N, ...) -> (N, T, ...)
        self.obs = np.swapaxes(rollout_obs, 0, 1)
        self.actions = np.swapaxes(rollout_actions, 0, 1)
        self.log_probs = np.swapaxes(rollout_log_probs, 0, 1)
        self.returns = np.swapaxes(rollout_returns, 0, 1)
        self.advantages = np.swapaxes(rollout_advantages, 0, 1)
        self.dones = np.swapaxes(rollout_dones, 0, 1)
        self.lengths = np.swapaxes(rollout_lengths, 0, 1)

        self.h_states = np.transpose(rollout_h_states, (2, 0, 1, 3))
        self.c_states = np.transpose(rollout_c_states, (2, 0, 1, 3))

        # Cache data shapes
        self.obs_shape = (self.obs.shape[-1],)
        self.action_shape = self.actions.shape[2:]

        # --- 2. Find all valid sequence start indices ---
        # A start (env_idx, step_idx) is valid if the sequence
        # [step_idx, step_idx + sequence_length) does not cross
        # an episode boundary (a 'done' signal).
        # --- 2. Find all valid sequence start indices AND their lengths ---
        # This now includes short sequences.
        self.valid_sequences = []
        for env_idx in range(self.n_envs):
            # Find all episode boundaries for this environment
            done_indices = np.where(self.dones[env_idx])[0]

            # List of (start, end) tuples for each episode
            ep_starts = [0] + (done_indices + 1).tolist()
            ep_ends = (done_indices + 1).tolist() + [self.n_steps]

            for ep_start, ep_end in zip(ep_starts, ep_ends):
                episode_len = ep_end - ep_start
                if episode_len == 0:
                    continue

                # Instead of a sliding window, we create mutually exclusive
                # sequences by "tiling" the episode.
                for step_idx in range(ep_start, ep_end, self.sequence_length):
                    # The sequence starts at step_idx
                    # The episode ends at ep_end
                    # The max end for this sequence is step_idx + sequence_length

                    # The actual end is the minimum of the max end and the episode end
                    actual_end = min(step_idx + self.sequence_length, ep_end)
                    actual_len = actual_end - step_idx

                    # We must have at least one step
                    if actual_len > 0:
                        self.valid_sequences.append(
                            (env_idx, step_idx, actual_len)
                        )

        self.total_valid_sequences = len(self.valid_sequences)
        self.batch_size = self.total_valid_sequences // self.n_minibatches
        self.sampler_length = self.n_minibatches
        assert self.sampler_length > 0, \
            (f"Warning: Sampler length is 0. "
             f"total_valid_sequences ({self.total_valid_sequences}) < num_sequences_per_batch "
             f"({self.batch_size}).")

    def __iter__(self):
        """
        Create a generator that yields mini-batches, compatible with multi-worker loading.
        """

        # --- 1. Get all possible sequence start indices and shuffle them ---
        # This is the main list of "work" to be done.
        indices = np.random.permutation(self.total_valid_sequences)

        # --- 2. Determine which batches this worker should process ---
        num_seqs = self.batch_size
        total_batches = self.total_valid_sequences // num_seqs

        # Get a list of all batch start points in the 'indices' array
        all_batch_starts = list(range(0, total_batches * num_seqs, num_seqs))

        # Shuffle the *order* of the batches
        np.random.shuffle(all_batch_starts)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Single-process
            batches_for_this_worker = all_batch_starts
        else:  # Multi-process: split the list of batches
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            per_worker = int(np.ceil(total_batches / float(num_workers)))
            start = worker_id * per_worker
            end = min(start + per_worker, total_batches)
            batches_for_this_worker = all_batch_starts[start:end]

        # --- 3. Yield batches assigned to this worker ---
        for start_idx in batches_for_this_worker:
            end_idx = start_idx + num_seqs

            # Get the (env, step, len) info for this mini-batch
            batch_indices = indices[start_idx:end_idx]
            seq_info = [self.valid_sequences[idx] for idx in batch_indices]

            batch_env_idxs = [info[0] for info in seq_info]
            batch_start_idxs = [info[1] for info in seq_info]
            batch_actual_lens = [info[2] for info in seq_info]

            # --- 3. Create padded buffers ---
            # (This part is identical to your original __iter__)
            max_sequence_length = self.sequence_length * TOTAL_SIZE
            obs_batch = np.zeros(
                (max_sequence_length, num_seqs, *self.obs_shape),
                dtype=self.obs.dtype
            )
            actions_batch = np.zeros(
                (max_sequence_length, num_seqs, *self.action_shape),
                dtype=self.actions.dtype
            )
            log_probs_batch = np.zeros(
                (max_sequence_length, num_seqs, 1),
                dtype=self.log_probs.dtype
            )
            returns_batch = np.zeros(
                (max_sequence_length, num_seqs, 1),
                dtype=self.returns.dtype
            )
            advantages_batch = np.zeros(
                (max_sequence_length, num_seqs, 1),
                dtype=self.advantages.dtype
            )
            mask_batch = np.zeros(
                (max_sequence_length, num_seqs, 1),
                dtype=bool
            )
            padding_mask_batch = np.zeros(
                (max_sequence_length, num_seqs, 1),
                dtype=bool
            )

            # --- 4. Fetch initial hidden states (no padding needed) ---
            h_starts = np.stack(
                [self.h_states[n, t] for n, t in zip(batch_env_idxs, batch_start_idxs)],
                axis=1
            )
            c_starts = np.stack(
                [self.c_states[n, t] for n, t in zip(batch_env_idxs, batch_start_idxs)],
                axis=1
            )

            # --- 5. Fill buffers with data and create mask ---
            for seq_i in range(num_seqs):
                n, t, L = batch_env_idxs[seq_i], batch_start_idxs[seq_i], batch_actual_lens[seq_i]

                # (This logic is identical to your original __iter__)
                padded_obs_batch = self.obs[n, t: t + L]
                unpadded_action_batch = self.actions[n, t: t + L]
                unpadded_log_probs_batch = self.log_probs[n, t: t + L].reshape(-1, 1)
                unpadded_returns_batch = self.returns[n, t: t + L]
                unpadded_advantages_batch = self.advantages[n, t: t + L]
                unpadded_lengths_batch = self.lengths[n, t: t + L]

                obs_mask = np.arange(TOTAL_SIZE) < unpadded_lengths_batch
                flattened_obs_batch = padded_obs_batch[obs_mask]
                obs_real_L = flattened_obs_batch.shape[0]
                obs_batch[:obs_real_L, seq_i] = flattened_obs_batch

                pad_mask = unpadded_lengths_batch.flatten().cumsum() - 1
                actions_batch[pad_mask, seq_i] = unpadded_action_batch
                log_probs_batch[pad_mask, seq_i] = unpadded_log_probs_batch
                returns_batch[pad_mask, seq_i] = unpadded_returns_batch
                advantages_batch[pad_mask, seq_i] = unpadded_advantages_batch
                mask_batch[pad_mask, seq_i] = True
                padding_mask_batch[:, seq_i, 0] = np.arange(max_sequence_length) < obs_real_L

            # --- 6. Yield the CPU tensors ---
            # (This is identical to your original __iter__ post-edit)
            yield (
                torch.from_numpy(obs_batch).transpose(0, 1),
                torch.from_numpy(actions_batch).transpose(0, 1),
                torch.from_numpy(log_probs_batch).transpose(0, 1),
                torch.from_numpy(returns_batch).transpose(0, 1),
                torch.from_numpy(advantages_batch).transpose(0, 1),
                torch.from_numpy(mask_batch).transpose(0, 1),
                torch.from_numpy(padding_mask_batch).transpose(0, 1),
                torch.from_numpy(h_starts),  # No transpose
                torch.from_numpy(c_starts),  # No transpose
            )


def pack_obs(batch_obs, lengths):
    if isinstance(lengths, np.ndarray):
        lengths = torch.from_numpy(lengths).long()
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    unsorted_idx = sorted_idx.argsort()
    sorted_obs = batch_obs[sorted_idx]
    packed_obs = pack_padded_sequence(sorted_obs, lengths_sorted, batch_first=True, enforce_sorted=True)
    return packed_obs, sorted_idx, unsorted_idx
