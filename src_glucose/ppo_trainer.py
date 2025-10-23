import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import os
from gymnasium.vector import SyncVectorEnv
from torch.distributions import Beta
from stable_baselines3.common.distributions import sum_independent_dims
from functools import partial
import random

INSULIN_ACTION_LOW = 0.0
INSULIN_ACTION_HIGH = 2.0


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


class MiniGridEncoder(nn.Module):
    """
    A feature extractor that uses self-attention on the input features.

    This network treats the input features as a sequence and uses a
    Multi-Head Attention block to learn the relationships between them.

    :param input_dim: The input dimension.
    :param hidden_dim: The final output dimension of the extractor.
    :param embed_dim: The dimension to embed each input feature into.
    :param num_heads: The number of attention heads. Must be a divisor of embed_dim.
    """

    def __init__(
            self,
            input_dim,
            hidden_dim: int = 128,
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

    def __init__(self, action_dim: int, low: float = 0.0, high: float = 2.0):
        super().__init__()
        self.action_dim = action_dim
        self.scale = float(high - low)
        self.bias = float(low)
        self.epsilon = 1e-6
        self.distribution = None
        self.alpha = None
        self.beta = None
        self.logit_clamp_min = -5.0
        self.logit_clamp_max = 5.0

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
        self.lstm = nn.LSTM(encoder_output_dim, hidden_dim, batch_first=True)
        self.actor_head = nn.Linear(hidden_dim, action_dim * 2)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.dist = CustomBetaDistribution(
            action_dim,
            low=INSULIN_ACTION_LOW,
            high=INSULIN_ACTION_HIGH,
        )

    def forward(self, x, hidden_state=None, deterministic=False):
        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)

        if is_packed:
            # If the input is a packed sequence, apply the encoder to the data part
            encoded_data = self.encoder(x.data)
            # Create a new packed sequence with the encoded data
            lstm_input = torch.nn.utils.rnn.PackedSequence(encoded_data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
        else:
            # Handle non-packed sequences (e.g., during prediction/evaluation)
            batch_size, seq_len, feature_dim = x.shape
            # Reshape for the encoder: (batch * seq_len, features)
            x_reshaped = x.reshape(batch_size * seq_len, feature_dim)
            # Encode the features
            encoded_x = self.encoder(x_reshaped)
            # Reshape back for the LSTM: (batch, seq_len, encoder_output_dim)
            lstm_input = encoded_x.reshape(batch_size, seq_len, -1)

        lstm_out, new_hidden = self.lstm(lstm_input, hidden_state)

        if is_packed:
            lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            batch_size = lstm_out.size(0)
            last_step_indices = lengths.long() - 1
            last_outputs = lstm_out[torch.arange(batch_size), last_step_indices, :]
        else:
            last_outputs = lstm_out[:, -1, :]

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
        return (torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim))


class RecurrentPPO:
    def __init__(self, env, env_name=None, hidden_dim=128, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 entropy_coef=0.01, vf_coef=0.5, clip_range=0.2, seed=None,
                 batch_size=64, n_steps=2048, n_epochs=10, eval_freq=10000, eval_episodes=500,
                 log_dir="../logs_glucose/ppo_logs", env_creator_fn=None):
        assert isinstance(env.action_space, gym.spaces.Box), "Continuous action spaces only."

        self.env = env
        self.env_name = env_name
        self.env_creator_fn = env_creator_fn if env_creator_fn is not None else partial(gym.make, env_name)
        self.input_dim = env.observation_space.shape[-1]
        self.action_dim = env.action_space.shape[0]
        self.hidden_dim = hidden_dim

        self.seed = seed
        if seed is not None:
            self.set_random_seed(seed)

        self.action_scale = torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.clip_range = clip_range
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.log_dir = log_dir
        self.last_obs = None
        self.last_hidden_state = None

        self._num_timesteps = 0
        self.best_mean_reward = -np.inf
        self.mean_ep_length = 0.0

        # Create log directory if it doesn't exist
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        self.encoder = MiniGridEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
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
            f"Checkpoint loaded from {path}. Resuming at timestep {new_agent._num_timesteps} with best reward {new_agent.best_mean_reward:.2f} and mean length {new_agent.mean_ep_length:.2f}.")

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
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                # Use the mode for deterministic prediction
                action, _, new_hidden_state = self.ac_network(obs_tensor, hidden_state, deterministic=True)
            else:
                # Sample for stochastic action
                dist, _, new_hidden_state = self.ac_network(obs_tensor, hidden_state, deterministic=False)
                action = dist.sample()

        # The environment expects a flattened numpy array
        action = action.cpu().numpy().flatten()
        return action, new_hidden_state

    def _compute_advantages_and_returns(self, rewards, values, dones, steps_taken):
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_advantage = 0
        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(self.last_obs).unsqueeze(0)
            _, last_value, _ = self.ac_network(last_obs_tensor, self.last_hidden_state)
            last_value = last_value.item()

        for t in reversed(range(n_steps)):
            mask = 1.0 - dones[t]
            effective_gamma = self.gamma ** steps_taken[t]
            delta = rewards[t] + effective_gamma * last_value * mask - values[t]
            last_advantage = delta + effective_gamma * self.gae_lambda * last_advantage * mask
            advantages[t] = last_advantage
            last_value = values[t]

        returns = advantages + values
        return advantages, returns

    def _evaluate_policy(self, n_eval_envs=4, n_eval_episodes=None):
        n_eval_episodes = n_eval_episodes or self.eval_episodes
        eval_env = SyncVectorEnv([lambda: self.env_creator_fn() for _ in range(n_eval_envs)])
        all_episode_rewards = []
        all_episode_steps = []
        episode_rewards = np.zeros(n_eval_envs)
        episode_steps = np.zeros(n_eval_envs)

        # Seed each parallel environment with a unique, deterministic seed
        if self.seed is not None:
            eval_seeds = [self.seed + i for i in range(n_eval_envs)]
            obs, _ = eval_env.reset(seed=eval_seeds)
        else:
            obs, _ = eval_env.reset()

        hidden_state = self.ac_network.init_hidden_state(batch_size=n_eval_envs)

        while len(all_episode_rewards) < n_eval_episodes:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(1)
            with torch.no_grad():
                action_env, _, hidden_state = self.ac_network(obs_tensor, hidden_state, deterministic=True)
            obs, reward, term, trunc, _ = eval_env.step(action_env.numpy())
            dones = term | trunc
            episode_rewards += reward
            episode_steps += 1
            for i, done in enumerate(dones):
                if done:
                    all_episode_rewards.append(episode_rewards[i])
                    all_episode_steps.append(episode_steps[i])
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                    hidden_state[0][:, i, :] = 0
                    hidden_state[1][:, i, :] = 0
        eval_env.close()
        return (np.mean(all_episode_rewards[:n_eval_episodes]),
                {'ep_length': np.mean(all_episode_steps[:n_eval_episodes])})

    def fit(self, total_timesteps):
        if self.seed is not None:
            obs, _ = self.env.reset(seed=self.seed)
        else:
            obs, _ = self.env.reset()

        hidden_state = self.ac_network.init_hidden_state()

        start_timesteps = self._num_timesteps
        target_timesteps = start_timesteps + total_timesteps

        updates = 0
        model_save_num = 1
        next_eval = (start_timesteps // self.eval_freq + 1) * self.eval_freq

        while self._num_timesteps < target_timesteps:
            updates += 1
            rollout_obs, rollout_rewards, rollout_dones = [], [], []
            rollout_actions, rollout_log_probs, rollout_values = [], [], []
            rollout_h_states, rollout_c_states = [], []
            rollout_steps_taken = []

            current_episode_reward = 0
            current_episode_length = 0
            ep_rewards_this_rollout = []
            ep_lengths_this_rollout = []

            for _ in range(self.n_steps):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                h_x, c_x = hidden_state
                rollout_h_states.append(h_x.detach().squeeze())
                rollout_c_states.append(c_x.detach().squeeze())
                with torch.no_grad():
                    dist, value, hidden_state = self.ac_network(obs_tensor, hidden_state)

                action_env = dist.last_sampled_action
                log_prob = dist.log_prob(action_env).sum(axis=-1)

                # action_env = dist.sample()
                # log_prob = dist.log_prob(action_env, gaussian_actions=dist.gaussian_actions).sum(axis=-1)
                new_obs, reward, term, trunc, info = self.env.step(action_env.numpy().flatten())
                done = term or trunc
                steps_taken = info.get('steps_taken', 1)
                self._num_timesteps += steps_taken
                current_episode_reward += reward
                current_episode_length += steps_taken
                rollout_obs.append(obs)
                rollout_actions.append(action_env.numpy().flatten())
                rollout_rewards.append(reward)
                rollout_dones.append(done)
                rollout_log_probs.append(log_prob.item())
                rollout_values.append(value.item())
                rollout_steps_taken.append(steps_taken)
                obs = new_obs
                if done:
                    ep_rewards_this_rollout.append(current_episode_reward)
                    ep_lengths_this_rollout.append(current_episode_length)
                    current_episode_reward = 0
                    current_episode_length = 0
                    obs, _ = self.env.reset()
                    hidden_state = self.ac_network.init_hidden_state()

            self.last_obs = obs
            self.last_hidden_state = hidden_state

            advantages, returns = self._compute_advantages_and_returns(rollout_rewards, rollout_values, rollout_dones,
                                                                       rollout_steps_taken)

            # *** KEY CHANGE: NORMALIZE ADVANTAGES ***
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            flat_advantages = torch.FloatTensor(advantages)
            flat_returns = torch.FloatTensor(returns)
            flat_actions = torch.FloatTensor(np.array(rollout_actions))
            flat_log_probs = torch.FloatTensor(np.array(rollout_log_probs))
            flat_h_states = torch.stack(rollout_h_states)
            flat_c_states = torch.stack(rollout_c_states)
            total_transitions = self.n_steps

            policy_losses, value_losses, entropy_losses, approx_kls = [], [], [], []
            for _ in range(self.n_epochs):
                indices = np.arange(total_transitions)
                np.random.shuffle(indices)
                for start in range(0, total_transitions, self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    batch_obs_seqs = [rollout_obs[i] for i in batch_indices]
                    batch_actions = flat_actions[batch_indices]
                    batch_log_probs = flat_log_probs[batch_indices]
                    batch_advantages = flat_advantages[batch_indices]
                    batch_returns = flat_returns[batch_indices]
                    batch_h_state = flat_h_states[batch_indices].unsqueeze(0)
                    batch_c_state = flat_c_states[batch_indices].unsqueeze(0)
                    lengths = torch.LongTensor([len(o) for o in batch_obs_seqs])
                    sorted_lengths, sorted_idx = lengths.sort(descending=True)
                    padded_obs = torch.nn.utils.rnn.pad_sequence(
                        [torch.FloatTensor(batch_obs_seqs[i]) for i in sorted_idx], batch_first=True)
                    packed_obs = torch.nn.utils.rnn.pack_padded_sequence(padded_obs, sorted_lengths, batch_first=True,
                                                                         enforce_sorted=True)
                    sorted_actions = batch_actions[sorted_idx]
                    sorted_log_probs = batch_log_probs[sorted_idx]
                    sorted_advantages = batch_advantages[sorted_idx]
                    sorted_returns = batch_returns[sorted_idx]
                    sorted_h_state = batch_h_state[:, sorted_idx, :]
                    sorted_c_state = batch_c_state[:, sorted_idx, :]
                    new_dist, new_values, _ = self.ac_network(packed_obs, (sorted_h_state, sorted_c_state))
                    new_values = new_values.squeeze()
                    new_log_probs = new_dist.log_prob(sorted_actions).sum(-1)
                    ratio = (new_log_probs - sorted_log_probs).exp()
                    surr1 = ratio * sorted_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * sorted_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(new_values, sorted_returns)
                    entropy_loss = -new_dist.entropy().mean()
                    loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * entropy_loss

                    with torch.no_grad():
                        log_ratio = new_log_probs - sorted_log_probs
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    approx_kls.append(approx_kl)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), 0.5)
                    self.optimizer.step()

            mean_reward = np.mean(ep_rewards_this_rollout) if ep_rewards_this_rollout else np.nan
            mean_ep_length = np.mean(ep_lengths_this_rollout) if ep_lengths_this_rollout else np.nan
            print(
                f"Timestep: {self._num_timesteps}/{target_timesteps} | Mean Ep. Length {mean_ep_length:.2f} | Mean Reward: {mean_reward:.2f} | Entropy: {np.mean(entropy_losses):.3f} | Value Loss: {np.mean(value_losses):.3f} | Policy Loss: {np.mean(policy_losses):.3f} | Approx. KL {np.mean(approx_kls):.5f}")

            if self._num_timesteps >= next_eval:
                avg_eval_reward, info = self._evaluate_policy()
                print(f"--- EVALUATION at Timestep: {self._num_timesteps}/{target_timesteps} ---")
                print(f"Average deterministic reward: {avg_eval_reward:.2f} | Best reward: {self.best_mean_reward:.2f} | Mean Length: {self.mean_ep_length:.2f}\n")

                if avg_eval_reward > self.best_mean_reward:
                    self.best_mean_reward = avg_eval_reward
                    self.mean_ep_length = info['ep_length']
                    if self.log_dir:
                        save_path = os.path.join(self.log_dir,
                                                 f"best_model{model_save_num:02d}_{avg_eval_reward:.2f}.pth")
                        self.save_checkpoint(save_path)
                        print(f"*** New best model found and saved with reward: {self.best_mean_reward:.2f} and episode length {self.mean_ep_length:.2f} ***\n")
                        model_save_num += 1

                next_eval += self.eval_freq

        self.env.close()


class ContinuousSemiMDPWrapper(gym.Wrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self._gamma = gamma
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low[np.newaxis],
            high=env.observation_space.high[np.newaxis],
            shape=(1,) + env.observation_space.shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return np.expand_dims(obs, 0), info

    def step(self, action):
        steps_to_take = np.random.randint(1, 4)
        obs_lst, total_reward, actual_steps_taken = [], 0, 0
        term, trunc = False, False
        for i in range(steps_to_take):
            obs, reward, term, trunc, info = self.env.step(action)
            done = term or trunc
            obs_lst.append(obs)
            total_reward += (self._gamma ** i) * reward
            actual_steps_taken += 1
            if done:
                break
        info['steps_taken'] = actual_steps_taken
        return np.stack(obs_lst), total_reward, term, trunc, info


if __name__ == '__main__':
    env_name = 'Pendulum-v1'
    GAMMA = 0.9
    LOG_DIR = "../logs_glucose/ppo_logs"

    # --- Control training and loading ---
    DO_TRAINING = True
    LOAD_CHECKPOINT = False
    CHECKPOINT_TO_LOAD = os.path.join(LOG_DIR, "best_model.pth")
    DO_TESTING = True

    if DO_TRAINING:
        print("--- STARTING TRAINING ---")
        base_env = gym.make(env_name, max_episode_steps=200)
        env = ContinuousSemiMDPWrapper(base_env, gamma=GAMMA)

        agent = RecurrentPPO(env, env_name=env_name, gamma=GAMMA,
                             n_steps=2048,
                             entropy_coef=0.01,
                             clip_range=0.2,
                             batch_size=64,
                             gae_lambda=0.95,
                             n_epochs=10,
                             learning_rate=1e-3,
                             eval_freq=100_000,
                             log_dir=LOG_DIR)

        if LOAD_CHECKPOINT:
            agent.load_checkpoint(CHECKPOINT_TO_LOAD)

        agent.fit(total_timesteps=2_000_000)

    if DO_TESTING:
        print("\n--- TESTING TRAINED AGENT ---")
        # Create a dummy env to initialize the agent
        test_env_for_agent = ContinuousSemiMDPWrapper(gym.make(env_name, max_episode_steps=200), gamma=GAMMA)
        trained_agent = RecurrentPPO(test_env_for_agent, env_name=env_name, gamma=GAMMA, log_dir=LOG_DIR)
        model_path = os.path.join(LOG_DIR, "best_model.pth")
        trained_agent.load_checkpoint(model_path)
        test_env_for_agent.close()

        # Create the actual environment for testing with rendering
        test_env = ContinuousSemiMDPWrapper(gym.make(env_name, max_episode_steps=200, render_mode='human'), gamma=GAMMA)
        obs, _ = test_env.reset()
        hidden_state = trained_agent.ac_network.init_hidden_state()

        total_reward = 0
        terminated, truncated = False, False

        for _ in range(200):  # Run for one episode
            action, hidden_state = trained_agent.predict(obs, hidden_state, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break

        print(f"Total reward during test: {total_reward}")
        test_env.close()
