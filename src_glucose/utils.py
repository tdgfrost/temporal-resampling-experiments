from typing import Union, Callable, Optional, Tuple
import numpy as np
import torch
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from collections import deque
import argparse
import inquirer
import os
import shutil
from scipy.stats import trimboth

from simglucose.simulation.env import bg_in_range_magni, early_termination_reward
from gym_wrappers import AGGREGATE_WINDOW_SIZE, INSULIN_SCALE, SAMPLE_TIME
from gymnasium.vector import AsyncVectorEnv
from ppo_trainer import RecurrentPPO


class RecurrentReplayBufferEnv:
    def __init__(self, env, buffer_size: int = 100000, sequence_length: int = 64, burn_in_length: int = 20):
        self.observations = {
            i: deque(maxlen=buffer_size) for i in range(3)
        }
        self.actions = {
            i: deque(maxlen=buffer_size) for i in range(3)
        }
        self.rewards = {
            i: deque(maxlen=buffer_size) for i in range(3)
        }
        self.dones = {
            i: deque(maxlen=buffer_size) for i in range(3)
        }
        self.sample_bool = {
            i: deque(maxlen=buffer_size) for i in range(3)
        }
        self.visible_states = {
            i: deque(maxlen=buffer_size) for i in range(3)
        }

        self.buffer_size = buffer_size
        self.env = env
        self._tensors_set = False
        self._device = None
        self.batch_size = 32
        self.decoy_interval = 0
        self.n_samples = 0
        self.segments = self.n_samples // self.batch_size
        self.dataset_IQR_return = None
        self.dataset_IQR_std = None
        self.dataset_IQR_n_episodes = None
        self.reward_mean = {i: None for i in range(3)}
        self.reward_std = {i: None for i in range(3)}
        self.max_sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        # This will store the valid start indices for each decoy interval
        self.sequence_info = {i: [] for i in range(3)}

    def __iter__(self):
        return iter(self.generate())

    def __len__(self):
        return self.segments

    def set_generate_params(self, device: str = 'cpu', batch_size: int = None, decoy_interval: int = None,
                            max_sequence_length: int = None, burn_in_length: int = None, epoch_fraction: float = 1.0):
        """
        Scans the buffer to identify all episodes and their lengths, preparing for sampling.
        """
        self.batch_size = batch_size if batch_size is not None else self.batch_size
        self.decoy_interval = decoy_interval if decoy_interval is not None else self.decoy_interval
        self.max_sequence_length = max_sequence_length if max_sequence_length is not None else self.max_sequence_length
        self.burn_in_length = burn_in_length if burn_in_length is not None else self.burn_in_length

        assert 0.0 < epoch_fraction <= 1.0, "epoch_fraction must be in the range (0, 1]"

        # --- CORE LOGIC CHANGE: Identify episodes instead of fixed-length sequences ---
        dones_data = self.dones[self.decoy_interval]
        if isinstance(dones_data, torch.Tensor):
            dones_np = dones_data.cpu().numpy()
        else:  # Handles list, deque, or numpy array
            dones_np = np.array(dones_data, dtype=bool)

        if not dones_np.any():
            self.n_samples = 0
            self.segments = 0
            return

        # Clear previous episode info for this interval
        self.sequence_info[self.decoy_interval] = []

        done_indices = np.where(dones_np)[0]

        last_ep_start_idx = 0
        if done_indices.size > 0:
            # --- Your efficient NumPy logic for finding terminated episodes ---
            start_indices = np.roll(done_indices, 1) + 1
            start_indices[0] = 0
            lengths = done_indices - start_indices + 1

            # --- Integrate sliding window logic into your loop ---
            for start_idx, length in zip(start_indices, lengths):
                if length <= self.max_sequence_length:
                    # Short episode: Add one sequence.
                    # (start_of_training, length_of_training, start_of_episode)
                    self.sequence_info[self.decoy_interval].append((start_idx, length, start_idx))
                else:
                    # Long episode: Use sliding window.
                    last_possible_start = start_idx + length - self.max_sequence_length
                    for seq_start in range(start_idx, last_possible_start + 1):
                        # (start_of_training, length_of_training, start_of_episode)
                        self.sequence_info[self.decoy_interval].append((seq_start, self.max_sequence_length, start_idx))

            # The next episode will start after the last 'done'
            last_ep_start_idx = done_indices[-1] + 1

        # --- ADDED: Handle the final, possibly unterminated episode ---
        final_buffer_len = len(dones_np)
        if last_ep_start_idx < final_buffer_len:
            ep_start_idx = last_ep_start_idx
            final_ep_len = final_buffer_len - last_ep_start_idx

            if final_ep_len <= self.max_sequence_length:
                # The final short episode.
                self.sequence_info[self.decoy_interval].append((ep_start_idx, final_ep_len, ep_start_idx))
            else:
                # The final long episode.
                last_possible_start = last_ep_start_idx + final_ep_len - self.max_sequence_length
                for seq_start in range(last_ep_start_idx, last_possible_start + 1):
                    self.sequence_info[self.decoy_interval].append((seq_start, self.max_sequence_length, ep_start_idx))

        # n_samples is now the number of sequences in the buffer
        self.n_samples = len(self.sequence_info[self.decoy_interval])
        if self.n_samples == 0:
            self.segments = 0
        else:
            total_segments = self.n_samples // self.batch_size
            if total_segments == 0:
                self.segments = 0
            else:
                # Calculate segments based on the fraction
                self.segments = int(total_segments * epoch_fraction)
                # Ensure we always have at least one segment if data is available
                if self.segments == 0:
                    self.segments = 1

        if self.segments == 0:
            print(f"Warning: Not enough episodes for sampling. "
                  f"Found {self.n_samples} episodes but batch size is {self.batch_size}.")

        self.set_to_tensors(device)

    def generate(self):
        """
        Generates batches by sampling EPISODE indices and then padding them.
        """
        assert self.segments > 0, "Not enough episodes to generate a batch. Call set_generate_params()."

        # Sample indices corresponding to the episodes, not timesteps
        episode_indices = np.random.choice(
            self.n_samples,
            size=(self.segments, self.batch_size),
            replace=False
        )

        idxs_tensor = torch.tensor(episode_indices, dtype=torch.long, device=self._device)

        for batch_of_episode_indices in idxs_tensor:
            yield self.fetch_transition_batch(idxs=batch_of_episode_indices, decoy_interval=self.decoy_interval)

    def fetch_transition_batch(self, idxs: torch.Tensor, decoy_interval: int = 0):
        """
        Fetches data for a batch of sequences, pads them to max_sequence_length,
        and returns the padded tensors along with a mask.
        """
        assert self._tensors_set, "Replay buffer must be set to tensors first."

        batch_size = len(idxs)
        # NEW: Total length is burn-in + max sequence length
        max_train_len = self.max_sequence_length
        burn_in_len = self.burn_in_length
        max_fetch_len = burn_in_len + max_train_len

        obs_shape = self.observations[decoy_interval][0].shape

        # Create buffers with the new max_fetch_len
        obs_batch = torch.zeros((batch_size, max_fetch_len, *obs_shape), dtype=torch.float32, device=self._device)
        next_obs_batch = torch.zeros_like(obs_batch)
        action_batch = torch.zeros((batch_size, max_fetch_len, 1), dtype=torch.float32, device=self._device)
        reward_batch = torch.zeros((batch_size, max_fetch_len, 1), dtype=torch.float32, device=self._device)
        done_batch = torch.zeros((batch_size, max_fetch_len, 1), dtype=torch.bool, device=self._device)
        visible_batch = torch.zeros((batch_size, max_fetch_len, 1), dtype=torch.bool, device=self._device)

        # 1. Get info from the sampled indices
        seq_info_batch = self.sequence_info[decoy_interval][idxs]
        train_starts = seq_info_batch[:, 0]
        actual_train_lens = seq_info_batch[:, 1]
        ep_starts = seq_info_batch[:, 2]

        # 2. Calculate burn-in and fetch indices
        # Find the real start of data to fetch (clipping at episode start)
        fetch_starts = torch.max(ep_starts, train_starts - burn_in_len)
        # Calculate how many burn-in steps we *actually* got
        actual_burn_in_lens = (train_starts - fetch_starts).long()
        # Calculate total length of data to fetch
        total_fetch_lens = actual_burn_in_lens + actual_train_lens

        # 3. Create index grids. We will right-align all sequences.
        base_indices = torch.arange(max_fetch_len, device=self._device).unsqueeze(0)
        # Mask for all valid data (burn-in + train)
        # Data is valid from index 0 up to total_fetch_lens
        padding_mask = base_indices < total_fetch_lens.unsqueeze(1)

        # 4. Create index grids for fetching all data in one go
        # This calculates the buffer index for each position in the output tensor
        indices = (fetch_starts.unsqueeze(1) + base_indices)
        next_indices = indices + 1

        # 5. Fetch all data using the masks and indices
        obs_batch[padding_mask] = self.observations[decoy_interval][indices[padding_mask]]
        action_batch[padding_mask] = self.actions[decoy_interval][indices[padding_mask]].unsqueeze(-1)
        reward_batch[padding_mask] = self.rewards[decoy_interval][indices[padding_mask]].unsqueeze(-1)
        done_batch[padding_mask] = self.dones[decoy_interval][indices[padding_mask]].unsqueeze(-1)
        visible_batch[padding_mask] = self.visible_states[decoy_interval][indices[padding_mask]].unsqueeze(-1)

        # 6. --- CORRECTED NEXT_OBS LOGIC ---
        next_obs_batch[padding_mask] = self.observations[decoy_interval][next_indices[padding_mask]]

        # 7. Create the TRAINING mask (excludes padding AND burn-in)
        # Training data starts *after* the actual burn-in
        train_mask_start_idx = actual_burn_in_lens.unsqueeze(1)
        # Mask is True from the start index up to the end of valid data (handled by padding_mask)
        train_mask_bool = (base_indices >= train_mask_start_idx) & padding_mask

        # Add dimension to masks to match data [B, S, 1]
        padding_mask = padding_mask.unsqueeze(-1)
        train_mask = train_mask_bool.unsqueeze(-1)

        # Get the next_padding_mask for next_obs
        next_padding_mask = torch.roll(padding_mask, shifts=-1, dims=1)
        next_padding_mask[:, -1, :] = False  # Last step does not have a next step

        # This can be derived from the visible_batch after fetching
        next_visible_batch = torch.roll(visible_batch, shifts=-1, dims=1)
        next_visible_batch[:, -1] = False  # Last step does not have a next step

        # Get our lengths
        lengths = padding_mask.sum(dim=1).squeeze(-1).cpu().to(torch.int64).clamp(min=1)
        next_lengths = next_padding_mask.sum(dim=1).squeeze(-1).cpu().to(torch.int64).clamp(min=1)

        return (obs_batch, action_batch, reward_batch, next_obs_batch, done_batch,
                visible_batch, next_visible_batch, padding_mask, next_padding_mask, train_mask,
                lengths, next_lengths)

    def save(self, path: str):
        if os.path.exists(path):
            print(f"Warning: Overwriting existing replay buffer at {path}")
            shutil.rmtree(path)
        os.makedirs(path)
        for key, value in [
            ('observations', self.observations),
            ('actions', self.actions),
            ('rewards', self.rewards),
            ('dones', self.dones),
            ('sample_bool', self.sample_bool),
            ('visible_states', self.visible_states)
        ]:
            save_dict = {i: list(value[i]) for i in range(3)}
            np.savez(os.path.join(path, f'{key}.npz'),
                     **{str(k): v for k, v in save_dict.items()})

        # Save dataset IQR return and scaling
        save_dict = {'dataset_iqr_avg': self.dataset_IQR_return, 'dataset_iqr_std': self.dataset_IQR_std,
                     'dataset_iqr_n_episodes': self.dataset_IQR_n_episodes}
        save_dict.update({f'reward_mean_{i}': self.reward_mean[i] for i in range(3)})
        save_dict.update({f'reward_std_{i}': self.reward_std[i] for i in range(3)})
        np.savez(os.path.join(path, 'rewards_scale.npz'), **save_dict)

        # Mark saving as complete
        with open(os.path.join(path, 'COMPLETE'), 'w') as f:
            f.close()

    def load(self, path: str):
        for key, value in [
            ('observations', self.observations),
            ('actions', self.actions),
            ('rewards', self.rewards),
            ('dones', self.dones),
            ('sample_bool', self.sample_bool),
            ('visible_states', self.visible_states)
        ]:
            loaded = np.load(os.path.join(path, f'{key}.npz'), allow_pickle=True)
            for k in loaded.files:
                value[int(k)] = deque(loaded[k], maxlen=self.buffer_size)

        # Load dataset IQR return and scaling
        loaded_scale = np.load(os.path.join(path, 'rewards_scale.npz'), allow_pickle=True)
        self.dataset_IQR_return = float(loaded_scale['dataset_iqr_avg'])
        self.dataset_IQR_std = float(loaded_scale['dataset_iqr_std'])
        self.dataset_IQR_n_episodes = int(loaded_scale['dataset_iqr_n_episodes'])
        for i in range(3):
            self.reward_mean[i] = float(loaded_scale[f'reward_mean_{i}'])
            self.reward_std[i] = float(loaded_scale[f'reward_std_{i}'])

        # Tweak to shorten to 1M steps if needed
        print('Trimming 10M to ~1M steps...')
        for i in range(3):
            n_episodes = np.where(np.array(self.dones[i]))[0].shape[0] // 10
            new_idx = np.where(np.array(self.dones[i]))[0][n_episodes] + 1
            self.observations[i] = deque(list(self.observations[i])[:new_idx + 1], maxlen=self.buffer_size)
            for arr in [self.actions, self.rewards, self.dones,
                        self.sample_bool, self.visible_states]:
                arr[i] = deque(list(arr[i])[:new_idx], maxlen=self.buffer_size)

    def reset(self, seed: int = None):
        obs, info = self.env.reset(seed=seed)
        ep_buffer = self._reset_ep_buffer(obs)
        return obs, info, ep_buffer

    @staticmethod
    def _reset_ep_buffer(obs):
        return {
            'all_obs': [obs],
            'all_action': [],
            'all_reward': [],
            'all_term': [],
            'all_trunc': [],
            'all_done': [],
            'visible_state': [True]
        }

    def fill_buffer(self, model, n_frames: int = 1_000, seed: int = None):
        with tqdm(total=n_frames, desc="Progress", mininterval=2.0) as pbar:
            frame_count = 0
            if seed is None:
                seed = 123

            obs, info, ep_buffer = self.reset(seed=seed)
            model.set_random_seed(seed)
            total_rewards = []

            while frame_count < n_frames:
                done = False
                lstm_states = model.ac_network.init_hidden_state(batch_size=1)
                total_reward = 0
                while not done:
                    action, lstm_states = model.predict(np.expand_dims(obs, 0), hidden_state=lstm_states, deterministic=True)
                    obs, reward, term, trunc, info = self.env.step(action)
                    # Get the real action delivered
                    real_action = obs[1] * INSULIN_SCALE
                    done = term or trunc
                    total_reward += reward

                    self.update_episode_buffer(obs, real_action, reward, term, trunc, info, ep_buffer)

                    if done:
                        ep_buffer['all_obs'] = ep_buffer['all_obs'][:-1]
                        obs, info = self.env.reset(seed=seed + frame_count)

                    pbar.update(1)
                    frame_count += 1
                    model.set_random_seed(seed + frame_count)

                # Add ep_buffer to permanent buffer
                self.update_permanent_buffer(ep_buffer)
                # Reset ep_buffer and add 'obs' to it
                ep_buffer = self._reset_ep_buffer(obs)

                # Keep track of total rewards for stats
                total_rewards.append(total_reward)

            # Add a garbage all-zeros "final obs"
            for i in range(3):
                self.observations[i] += [np.zeros_like(self.observations[0][0])]

            # Save the IQR dataset return
            total_rewards = np.array(total_rewards)
            IQR = trimboth(total_rewards, proportiontocut=0.25)

            self.dataset_IQR_return = IQR.mean()
            self.dataset_IQR_std = IQR.std()
            self.dataset_IQR_n_episodes = len(IQR)

            # Scale the rewards for training
            for i in [0, 1, 2]:
                all_rewards = np.array(self.rewards[i])
                r_mean, r_std = all_rewards.mean(), all_rewards.std()
                # Standardise rewards
                norm_rewards = (all_rewards - r_mean) / (r_std + 1e-8)
                self.rewards[i] = deque(norm_rewards.tolist(), maxlen=self.buffer_size)
                self.reward_mean[i] = r_mean
                self.reward_std[i] = r_std

    def set_to_tensors(self, device: str = 'cpu', i: int = None):
        def _set_device(some_arr):
            if isinstance(some_arr, torch.Tensor):
                return some_arr.to(device)
            return torch.from_numpy(np.array(some_arr)).to(device)

        i = self.decoy_interval if i is None else i

        self.observations[i] = _set_device(self.observations[i]).float()
        self.actions[i] = _set_device(self.actions[i]).float()
        self.rewards[i] = _set_device(self.rewards[i]).float()
        self.dones[i] = _set_device(self.dones[i]).bool()
        self.visible_states[i] = _set_device(self.visible_states[i]).bool()
        self.sequence_info[i] = _set_device(self.sequence_info[i]).long()

        self._tensors_set = True
        self._device = device

    @staticmethod
    def update_episode_buffer(obs, action: Union[int, np.ndarray], reward: float, term: bool, trunc: bool, info: dict,
                              ep_buffer: dict):
        ep_buffer['all_obs'] += [obs]
        ep_buffer['all_action'] += [action]
        ep_buffer['all_reward'] += [reward]
        ep_buffer['all_term'] += [term]
        ep_buffer['all_trunc'] += [trunc]
        ep_buffer['all_done'] += [term or trunc]
        ep_buffer['visible_state'] += [info['steps_until_action_available'] == 0]

    def update_permanent_buffer(self, ep_buffer: dict):
        visible_idxs = ep_buffer['visible_state']
        for i in range(2):
            for arr, key in [
                (self.observations, 'obs'), (self.actions, 'action'), (self.rewards, 'reward'), (self.dones, 'done')
            ]:
                arr[i] += ep_buffer[f'all_{key}']

            self.sample_bool[i] += [True for _ in range(len(ep_buffer[f'all_done']))]

        # Update visible states
        self.visible_states[0] += visible_idxs
        self.visible_states[1] += [True for _ in range(len(ep_buffer[f'all_done']))]

        # Generate our aggregated datapoints for decoy_interval = 2
        # e.g., for window_size = 3, we use the following aggregates:
        # for i in [3, 6, 9, ...]
        #      [t0,  t1,  t2,  t3,  t4,  t5,  t6,  ...]
        # obs:  |----------|    |---------|    |----   (t0-t2, t3-t5, etc)
        # act:             |---------|    |-------     (t2-t4, t5-t7, etc)
        # rew:             |---------|    |-------     (t2-t4, t5-t7, etc)
        # (reward calculated from next obs)

        agg_window = AGGREGATE_WINDOW_SIZE

        obs_data = np.array(ep_buffer['all_obs'])
        action_data = np.array(ep_buffer['all_action'])
        done_data = np.array(ep_buffer['all_done'])
        reward_data = np.array(ep_buffer['all_reward'])

        # --- Aggregage obs ---
        obs_indices = range(agg_window, len(obs_data) + 1, agg_window)
        obs_splits = np.array_split(obs_data, obs_indices, axis=0)[:-1]  # Exclude last partial window
        obs = [np.mean(chunk, axis=0) for chunk in obs_splits if chunk.size > 0]

        # --- Aggregate actions / dones / rewards ---
        slice_indices = range(agg_window - 1, len(obs_data), agg_window)
        action_splits = np.array_split(action_data, slice_indices, axis=0)[1:]  # Exclude first partial window
        done_splits = np.array_split(done_data, slice_indices, axis=0)[1:]  # Exclude first partial window
        reward_splits = np.array_split(reward_data, slice_indices, axis=0)[1:]  # Exclude first partial window

        actions = [np.mean(chunk, axis=0) for chunk in action_splits if chunk.size > 0]
        dones = [np.any(chunk) for chunk in done_splits if chunk.size > 0]
        rewards = [np.mean(chunk) for chunk in reward_splits if chunk.size > 0]

        """
        # Calculate early terminations for the additional reward component
        terms = [
            np.any(ep_buffer['all_term'][idx: idx + agg_window])
            for idx in range(agg_window, len(ep_buffer['all_term']), agg_window)
        ]

        # Manually calculate reward
        bg_levels = np.array([
            np.mean(ep_buffer['all_obs'][idx: idx + agg_window], 0)[-3]
            for idx in range(agg_window, len(ep_buffer['all_obs']), agg_window)
        ]) * (600 - 10) + 10  # Scale back to real bg levels

        rewards = [bg_in_range_magni([i]) * SAMPLE_TIME for i in bg_levels]
        rewards[-1] += early_termination_reward(terms[-1])
        """

        if not dones or not dones[-1]:
            # Skip this episode - too short
            return
        self.observations[2] += obs
        self.actions[2] += actions
        self.rewards[2] += rewards
        self.dones[2] += dones
        self.sample_bool[2] += [True for _ in range(len(dones))]
        self.visible_states[2] += [True for _ in range(len(dones))]


class EnvironmentEvaluator:
    def __init__(self, env, n_trials: int, running_average_obs: bool = False):
        assert n_trials > 0, "n_trials must be positive"
        # Scale rewards to [0, 1]
        self.env = env
        self.n_trials = n_trials
        self.running_average_obs = running_average_obs

    def __call__(self, algo) -> float:
        mean_returns = []
        running_average_obs = deque(maxlen=AGGREGATE_WINDOW_SIZE)
        for _ in range(self.n_trials):
            obs, info = self.env.reset()
            running_average_obs.append(obs)
            done = False
            total_reward = 0.0
            hidden_state = algo.get_initial_states(batch_size=1)

            while not done:
                with torch.no_grad():
                    if self.running_average_obs:
                        obs = np.mean(np.stack(running_average_obs), axis=0)
                    action, hidden_state = algo.predict(np.expand_dims(obs, 0),
                                                        hidden_state=hidden_state,
                                                        deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                total_reward += reward

            mean_returns.append(total_reward)

        return float(np.mean(mean_returns)), float(np.std(mean_returns) / np.sqrt(self.n_trials))


class ParallelEnvironmentEvaluator:
    """
    Evaluates a policy over n_eval_episodes using parallel environments.

    :param env_fn: A function that returns a new Gymnasium environment instance.
    :param n_eval_episodes: The total number of episodes to evaluate.
    :param n_eval_envs: The number of parallel environments to run.
    :param running_average_obs: Whether to feed the policy a running average
        of observations.
    :param aggregate_window_size: The window size for the observation average.
    :param seed: An optional seed for environment reproducibility.
    """

    def __init__(self,
                 env_fn: Callable,
                 n_eval_episodes: int,
                 n_eval_envs: int,
                 running_average_obs: bool = False,
                 aggregate_window_size: int = AGGREGATE_WINDOW_SIZE,
                 seed: Optional[int] = None):

        assert n_eval_episodes > 0, "n_eval_episodes must be positive"
        assert n_eval_envs > 0, "n_eval_envs must be positive"

        self.env_fn = env_fn
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.running_average_obs = running_average_obs
        self.aggregate_window_size = aggregate_window_size
        self.seed = seed

    def __call__(self, algo, seed=None) -> Tuple[float, float]:
        # --- 1. Setup ---
        all_episode_rewards = []
        episode_rewards = np.zeros(self.n_eval_envs)
        seed = seed or self.seed

        # Create the vectorized environment
        self.eval_env = AsyncVectorEnv([self.env_fn for _ in range(self.n_eval_envs)])

        running_avg_deques = None
        if self.running_average_obs:
            running_avg_deques = [deque(maxlen=self.aggregate_window_size)
                                  for _ in range(self.n_eval_envs)]

        # --- 2. Reset Envs ---
        if seed is not None:
            # Seed each parallel environment deterministically
            eval_seeds = [int(seed + i) for i in range(self.n_eval_envs)]
            obs, info = self.eval_env.reset(seed=eval_seeds)
        else:
            obs, info = self.eval_env.reset()

        # Initialize running average deques with the first observation
        if self.running_average_obs:
            for i in range(self.n_eval_envs):
                running_avg_deques[i].append(obs[i])

        # Get initial hidden state for the batch
        hidden_state = algo.get_initial_states(batch_size=self.n_eval_envs)

        # --- 3. Run Episodes ---
        with tqdm(total=self.n_eval_episodes, desc="Evaluating Episodes", mininterval=2.0) as pbar:
            while len(all_episode_rewards) < self.n_eval_episodes:

                # --- 3a. Prepare Observations ---
                obs_to_predict = obs
                if self.running_average_obs:
                    # Calculate mean obs for each env in the batch
                    mean_obs_batch = [np.mean(np.stack(deque), axis=0)
                                      for deque in running_avg_deques]
                    obs_to_predict = np.stack(mean_obs_batch) # for seq dim

                # --- 3b. Predict Action ---
                with torch.no_grad():
                    # Make sure the seq_dim is present
                    obs_to_predict = np.expand_dims(obs_to_predict, axis=-2)
                    action, hidden_state = algo.predict(obs_to_predict,
                                                        hidden_state=hidden_state,
                                                        deterministic=False)

                # --- 3c. Step Environment ---
                # action is a batch (B, ...), so we pass it directly
                # (or convert to numpy if it's a tensor)
                try:
                    action_np = action.cpu().numpy()
                except AttributeError:
                    action_np = action  # Assume it's already numpy

                next_obs, reward, terminated, truncated, info = self.eval_env.step(action_np)
                dones = terminated | truncated

                episode_rewards += reward

                # --- 3d. Handle Dones and State Updates ---
                for i, done in enumerate(dones):
                    # Always append the *next* observation for the running average
                    if self.running_average_obs:
                        running_avg_deques[i].append(next_obs[i])

                    if done:
                        # An episode finished
                        all_episode_rewards.append(episode_rewards[i])
                        episode_rewards[i] = 0  # Reset reward accumulator

                        # Update the progress bar
                        pbar.update(1)

                        # Reset hidden state for this env (mimicking your example)
                        if isinstance(hidden_state, tuple):
                            # Handle LSTM-like states (h, c)
                            hidden_state[0][:, i, :] = 0
                            hidden_state[1][:, i, :] = 0
                        elif hidden_state is not None:
                            # Handle GRU-like states
                            hidden_state[:, i, :] = 0

                        # Reset the running average deque for this env
                        if self.running_average_obs:
                            running_avg_deques[i].clear()
                            # Add the new obs from the auto-reset
                            running_avg_deques[i].append(next_obs[i])

                            # The observation for the next loop
                obs = next_obs

        # Cleanup
        self.eval_env.close()

        # Ensure we only use the requested number of episodes
        final_rewards = all_episode_rewards[:self.n_eval_episodes]

        return final_rewards


class WISOPEEvaluator:
    """
    Evaluates using weighted IS OPE.
    """

    def __init__(self,
                 ppo_agent: RecurrentPPO,
                 dataset: RecurrentReplayBufferEnv,
                 decoy_interval: int = 0,
                 gamma: float = 0.99,
                 device: str = 'cpu'):

        self.ppo_agent = ppo_agent
        self.ppo_agent.ac_network.to(device)
        self.dataset = dataset
        self.gamma = gamma
        self.decoy_interval = decoy_interval
        self.device = device

    def __call__(self, algo, seed=None) -> Tuple[float, float]:
        """
        Performs Weighted Importance Sampling (WIS) Off-Policy Evaluation (OPE).

        This function iterates through all full episodes in the replay buffer,
        computes the per-episode importance ratio (rho) and the un-normalized
        discounted return (G), and returns the WIS estimate:

        V(pi_e) = [ Sum(rho_i * G_i) ] / [ Sum(rho_i) ]
        """
        # Get data from buffer (will already be as tensors)
        obs_data, act_data, rew_data, done_data, visible_data = self.unpack_dataset()

        # Iterate through episodes, storing log-ratios and returns
        log_rho_list = []
        return_list = []

        # Find all episode boundaries
        done_indices = torch.where(done_data)[0]
        assert done_data[-1], "The last transition must be done to ensure complete episodes."

        start_idx = 0
        with tqdm(total=len(done_indices), desc="Processing episodes for OPE", mininterval=2.0) as pbar:
            for end_idx in done_indices:
                # Slice the episode data
                ep_obs = obs_data[start_idx : end_idx + 1]
                ep_act = act_data[start_idx : end_idx + 1]
                ep_rew = rew_data[start_idx : end_idx + 1]
                ep_visible = visible_data[start_idx : end_idx + 1]

                ep_len = len(ep_rew)
                if ep_len == 0:
                    start_idx = end_idx + 1
                    continue

                # Calculate the episode return (G)
                # discounts = torch.pow(self.gamma, torch.arange(ep_len, device=self.device))
                # ep_return = torch.sum(ep_rew * discounts)
                ep_return = torch.sum(ep_rew)

                # Calculate the episode importance ratio (rho)
                ppo_hn, ppo_cn = self.ppo_agent.ac_network.init_hidden_state(batch_size=1)
                ppo_h_state = (ppo_hn.to(self.device), ppo_cn.to(self.device))
                offline_h_state = algo.get_initial_states(batch_size=1)
                with torch.no_grad():
                    # Get our log_probs
                    log_prob_b = self.ppo_agent.ac_network.evaluate_actions(ep_obs.unsqueeze(0),
                                                                            hidden_state=ppo_h_state,
                                                                            actions=ep_act.unsqueeze(0))

                    log_prob_e = algo.evaluate_actions(ep_obs.unsqueeze(0),
                                                       hidden_state=offline_h_state,
                                                       actions=ep_act.unsqueeze(0)).squeeze()

                    # Calculate the log-ratio for the whole episode
                    ep_log_rhos = log_prob_e - log_prob_b
                    # - Cancel non-visible states
                    ep_log_rhos = torch.where(ep_visible, ep_log_rhos, torch.zeros_like(ep_log_rhos))
                    ep_log_rho = ep_log_rhos.sum()

                    # Store log_rho and return
                    log_rho_list.append(ep_log_rho)
                    return_list.append(ep_return)

                    # Update start index for next loop
                    start_idx = end_idx + 1

                pbar.update(1)

        # Perform WIS calculation
        all_log_rhos = torch.stack(log_rho_list)
        all_returns = torch.stack(return_list)

        # Find the maximum log_rho and substract for numerical stability
        max_log_rho = torch.max(all_log_rhos)
        stabilised_log_rhos = all_log_rhos - max_log_rho
        stabilised_rhos = stabilised_log_rhos.exp()

        numerator = (stabilised_rhos * all_returns).sum()
        denominator = stabilised_rhos.sum()

        wis_estimate = numerator / (denominator + 1e-6)

        return wis_estimate.item()

    def unpack_dataset(self):
        self.dataset.set_to_tensors(self.device, self.decoy_interval)
        obs_data = self.dataset.observations[self.decoy_interval]
        act_data = self.dataset.actions[self.decoy_interval]
        rew_data = self.dataset.rewards[self.decoy_interval]
        done_data = self.dataset.dones[self.decoy_interval]
        visible_data = self.dataset.visible_states[self.decoy_interval]

        # Correct for the extra meaningless "final" obs
        if len(obs_data) == len(act_data) + 1:
            obs_data = obs_data[:-1]

        # Get reward scaling to unnormalise the returns
        r_mean = self.dataset.reward_mean[self.decoy_interval]
        r_std = self.dataset.reward_std[self.decoy_interval]
        rew_data = rew_data * (r_std + 1e-8) + r_mean

        return obs_data, act_data, rew_data, done_data, visible_data


def parse_bool(value, name: str = ''):
    if isinstance(value, str) and value.lower() in ['true', 'false']:
        return value.lower() == "true"
    elif isinstance(value, (int, float)) and value in [0, 1]:
        return bool(value)
    elif isinstance(value, bool):
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid value for {name}: '{value}'. Must be a boolean e.g., True or true.")


def choose_ppo_agent():
    # Use inquirer to let the user select a folder
    message = "Please select PPO agent using UP/DOWN/ENTER."
    options = os.listdir("../logs/ppo_minigrid_logs/historic_bests")
    options = [i for i in options if i.endswith('.zip')]
    question = [inquirer.List('option',
                              message=message,
                              choices=options)]
    answer = inquirer.prompt(question)
    return f"../logs/ppo_minigrid_logs/historic_bests/{answer['option']}"
