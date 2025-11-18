from functools import partial
from typing import Union, Callable, Optional, Tuple, List
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

from gym_wrappers import AGGREGATE_WINDOW_SIZE, INSULIN_SCALE, MASTER_SEED
from gymnasium.vector import AsyncVectorEnv


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

        # Set sequence info to numpy
        self.sequence_info[self.decoy_interval] = np.array(self.sequence_info[self.decoy_interval])

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

        # Record the device for later
        self._device = device

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

        for batch_of_episode_indices in episode_indices:
            yield self.fetch_transition_batch(idxs=batch_of_episode_indices, decoy_interval=self.decoy_interval)

    def fetch_transition_batch(self, idxs: np.ndarray, decoy_interval: int = 0):
        """
        Fetches data for a batch of sequences, pads them to max_sequence_length,
        and returns the padded tensors along with a mask.
        """
        batch_size = len(idxs)
        # NEW: Total length is burn-in + max sequence length
        max_train_len = self.max_sequence_length
        burn_in_len = self.burn_in_length
        max_fetch_len = burn_in_len + max_train_len

        obs_shape = self.observations[decoy_interval][0].shape

        # Create buffers with the new max_fetch_len
        obs_batch_np = np.empty((batch_size, max_fetch_len, *obs_shape), dtype=np.float32)
        next_obs_batch_np = np.empty_like(obs_batch_np)
        action_batch_np = np.empty((batch_size, max_fetch_len, 1), dtype=np.float32)
        reward_batch_np = np.empty((batch_size, max_fetch_len, 1), dtype=np.float32)
        done_batch_np = np.empty((batch_size, max_fetch_len, 1), dtype=np.bool)
        visible_batch_np = np.empty((batch_size, max_fetch_len, 1), dtype=np.bool)

        # 1. Get info from the sampled indices
        seq_info_batch = self.sequence_info[decoy_interval][idxs]
        train_starts = seq_info_batch[:, 0]
        actual_train_lens = seq_info_batch[:, 1]
        ep_starts = seq_info_batch[:, 2]

        # 2. Calculate burn-in and fetch indices
        # Find the real start of data to fetch (clipping at episode start)
        fetch_starts = np.maximum(ep_starts, train_starts - burn_in_len)
        # Calculate how many burn-in steps we *actually* got
        actual_burn_in_lens = train_starts - fetch_starts
        # Calculate total length of data to fetch
        total_fetch_lens = actual_burn_in_lens + actual_train_lens

        # 3. Create index grids. We will right-align all sequences.
        base_indices = np.expand_dims(np.arange(max_fetch_len), 0)
        # Mask for all valid data (burn-in + train)
        # Data is valid from index 0 up to total_fetch_lens
        padding_mask_np = base_indices < np.expand_dims(total_fetch_lens, 1)

        # 4. Create index grids for fetching all data in one go
        # This calculates the buffer index for each position in the output tensor
        indices_np = np.expand_dims(fetch_starts, 1) + base_indices
        next_indices_np = indices_np + 1

        # 5. Fetch all data using the masks and indices
        obs_batch_np[padding_mask_np] = self.observations[decoy_interval][indices_np[padding_mask_np]]
        action_batch_np[padding_mask_np] = np.expand_dims(self.actions[decoy_interval][indices_np[padding_mask_np]], -1)
        reward_batch_np[padding_mask_np] = np.expand_dims(self.rewards[decoy_interval][indices_np[padding_mask_np]], -1)
        done_batch_np[padding_mask_np] = np.expand_dims(self.dones[decoy_interval][indices_np[padding_mask_np]], -1)
        visible_batch_np[padding_mask_np] = np.expand_dims(self.visible_states[decoy_interval][indices_np[padding_mask_np]], -1)
        next_obs_batch_np[padding_mask_np] = self.observations[decoy_interval][next_indices_np[padding_mask_np]]

        # 7. Create the TRAINING mask (excludes padding AND burn-in)
        # Training data starts *after* the actual burn-in
        train_mask_start_idx = np.expand_dims(actual_burn_in_lens, -1)
        # Mask is True from the start index up to the end of valid data (handled by padding_mask)
        train_mask_bool = (base_indices >= train_mask_start_idx) & padding_mask_np

        # Add dimension to masks to match data [B, S, 1]
        padding_mask_np = np.expand_dims(padding_mask_np, -1)
        train_mask_np = np.expand_dims(train_mask_bool, -1)

        # Get the next_padding_mask for next_obs
        next_padding_mask_np = np.roll(padding_mask_np, shift=-1, axis=1)
        next_padding_mask_np[:, -1, :] = False  # Last step does not have a next step

        # This can be derived from the visible_batch after fetching
        next_visible_batch_np = np.roll(visible_batch_np, shift=-1, axis=1)
        next_visible_batch_np[:, -1] = False  # Last step does not have a next step

        # Set tensors to correct device
        all_arrs = (obs_batch_np, action_batch_np, reward_batch_np, next_obs_batch_np, done_batch_np,
                       visible_batch_np, next_visible_batch_np, padding_mask_np, next_padding_mask_np, train_mask_np)
        all_tensors_set = []
        for some_arr in all_arrs:
            all_tensors_set.append(torch.from_numpy(some_arr).to(self._device, non_blocking=True))

        return tuple(all_tensors_set)

    def save(self, path: str):
        if os.path.exists(path):
            print(f"Warning: Overwriting existing replay buffer at {path}")
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
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
                value[int(k)] = loaded[k]

        # Load dataset IQR return and scaling
        loaded_scale = np.load(os.path.join(path, 'rewards_scale.npz'), allow_pickle=True)
        self.dataset_IQR_return = float(loaded_scale['dataset_iqr_avg'])
        self.dataset_IQR_std = float(loaded_scale['dataset_iqr_std'])
        self.dataset_IQR_n_episodes = int(loaded_scale['dataset_iqr_n_episodes'])
        for i in range(3):
            self.reward_mean[i] = float(loaded_scale[f'reward_mean_{i}'])
            self.reward_std[i] = float(loaded_scale[f'reward_std_{i}'])

        # Tweak to shorten from 1M to 500k steps (if needed)
        print(f"Trimming 1M to ~500k steps for dataset...")
        for i in range(3):
            done_indices = np.where(self.dones[i])[0]
            n_episodes = done_indices.shape[0] // 2
            start_idx = 0
            end_idx = done_indices[n_episodes] + 1
            self.observations[i] = self.observations[i][start_idx:end_idx + 1]
            for arr in [self.actions, self.rewards, self.dones,
                        self.sample_bool, self.visible_states]:
                arr[i] = arr[i][start_idx:end_idx]

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
            total_lengths = []

            while frame_count < n_frames:
                done = False
                lstm_states = model.ac_network.init_hidden_state(batch_size=1)
                total_reward = 0
                while not done:
                    action, lstm_states = model.predict(np.expand_dims(obs, 0), hidden_state=lstm_states,
                                                        deterministic=False)
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
                    if len(total_rewards) >= 1:
                        pbar_dict = {'avg_episode_reward': f"{np.mean(total_rewards):.2f}",
                                     'avg_episode_IQM_reward': f"{trimboth(np.array(total_rewards), proportiontocut=0.25).mean():.2f}",
                                     'avg_episode_length': f"{np.mean(total_lengths):.2f}",
                                     'refresh': False}
                        pbar.set_postfix(**pbar_dict)
                    frame_count += 1
                    model.set_random_seed(seed + frame_count)

                # Keep track of total rewards for stats
                total_rewards.append(total_reward)
                total_lengths.append(len(ep_buffer['all_done']))

                # Add ep_buffer to permanent buffer
                self.update_permanent_buffer(ep_buffer)
                # Reset ep_buffer and add 'obs' to it
                ep_buffer = self._reset_ep_buffer(obs)

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

    def set_to_tensors(self, device: str = 'cpu', i: int = None, *args):
        def _set_device(some_arr):
            if isinstance(some_arr, torch.Tensor):
                return some_arr.to(device, non_blocking=True)
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

        if not dones or not dones[-1]:
            # Skip this episode - too short
            return
        self.observations[2] += obs
        self.actions[2] += actions
        self.rewards[2] += rewards
        self.dones[2] += dones
        self.sample_bool[2] += [True for _ in range(len(dones))]
        self.visible_states[2] += [True for _ in range(len(dones))]


class ParallelEnvironmentEvaluator:
    """
    Evaluates a policy using parallel environments, ensuring a specific number
    of episodes are collected per patient ID.

    :param env_creator_fn: A function that returns a new Gymnasium environment instance.
    :param n_eval_episodes_per_id: The number of episodes to evaluate PER patient ID.
                                   (If test_ids is None, this acts as total episodes).
    :param n_eval_envs: The number of parallel environments to run.
    :param test_ids: List of patient IDs to evaluate.
    """

    def __init__(self,
                 env_creator_fn: Callable,
                 n_eval_episodes_per_id: int,
                 n_eval_envs: int,
                 test_ids: Optional[List[str]] = None,
                 gamma: float = 0.99,
                 running_average_obs: bool = False,
                 aggregate_window_size: int = 10,  # Assuming default or imported constant
                 seed: Optional[int] = None,
                 verbose: bool = True):

        assert n_eval_episodes_per_id > 0, "n_eval_episodes_per_id must be positive"
        assert n_eval_envs > 0, "n_eval_envs must be positive"

        self.env_creator_fn = env_creator_fn
        self.n_eval_envs = n_eval_envs
        self.running_average_obs = running_average_obs
        self.aggregate_window_size = aggregate_window_size
        self.seed = seed
        self.gamma = gamma
        self.verbose = verbose

        self.test_ids = test_ids
        self.env_id_map = None

        # Set the target per ID directly from input
        self.target_episodes_per_id = n_eval_episodes_per_id

        if test_ids is not None:
            assert n_eval_envs % len(test_ids) == 0, \
                "n_eval_envs must be a multiple of the number of test_ids."

            self.n_ids = len(test_ids)
            self.envs_per_id = n_eval_envs // self.n_ids

            # Calculate total target based on per-id count
            self.total_target_episodes = self.target_episodes_per_id * self.n_ids

            self.eval_env = AsyncVectorEnv([
                # Assign patient IDs in blocks
                partial(self.env_creator_fn, patient_ids=test_ids[i // self.envs_per_id])
                for i in range(self.n_eval_envs)
            ])

            # Create the mapping: [0] -> 'id_A', [1] -> 'id_A', ..., [8] -> 'id_B'
            self.env_id_map = [test_ids[i // self.envs_per_id] for i in range(self.n_eval_envs)]

            if self.verbose:
                print(f"--- Evaluator: Running balanced evaluation for {self.n_ids} IDs ---")
                print(
                    f"--- Target: {self.target_episodes_per_id} episodes per ID (Total: {self.total_target_episodes}) ---")
        else:
            # Fallback: if no IDs provided, treat the input as the total count
            self.total_target_episodes = n_eval_episodes_per_id
            self.eval_env = AsyncVectorEnv([self.env_creator_fn for _ in range(self.n_eval_envs)])
            if self.verbose:
                print(f"--- Evaluator: Running (unbalanced) evaluation for {self.total_target_episodes} episodes ---")

    def __call__(self, algo, seed=None) -> Tuple[float, float]:
        # --- 1. Setup ---
        episode_rewards = np.zeros(self.n_eval_envs)
        seed = seed or self.seed

        # Dictionaries/lists to store results
        if self.test_ids is not None:
            rewards_per_id = {id: [] for id in self.test_ids}
            do_balanced_eval = True
        else:
            all_episode_rewards = []
            do_balanced_eval = False

        try:
            original_device = algo._device
            algo.to('cpu')
            algo._device = 'cpu'
        except AttributeError:
            original_device = None

        running_avg_deques = None
        if self.running_average_obs:
            running_avg_deques = [deque(maxlen=self.aggregate_window_size)
                                  for _ in range(self.n_eval_envs)]

        # --- 2. Reset Envs ---
        if seed is not None:
            rng = np.random.default_rng(seed)  # Used local seed instead of global MASTER_SEED for safety
            eval_seeds = rng.integers(low=0, high=2 ** 32 - 1, size=self.n_eval_envs).tolist()
            obs, info = self.eval_env.reset(seed=eval_seeds)
        else:
            obs, info = self.eval_env.reset()

        if self.running_average_obs:
            for i in range(self.n_eval_envs):
                running_avg_deques[i].append(obs[i])

        try:
            hidden_state = algo.get_initial_states(batch_size=self.n_eval_envs)
        except:
            hidden_state = None

        # --- 3. Define Loop Condition ---
        if do_balanced_eval:
            def is_evaluation_done():
                # Loop until ALL test IDs have reached their target per ID
                return all(len(rewards_per_id[id]) >= self.target_episodes_per_id for id in self.test_ids)
        else:
            def is_evaluation_done():
                return len(all_episode_rewards) >= self.total_target_episodes

        pbar_total = self.total_target_episodes

        # --- 4. Run Episodes ---
        with tqdm(total=pbar_total, desc="Evaluating Episodes", mininterval=2.0, disable=not self.verbose,
                  leave=False) as pbar:
            while not is_evaluation_done():

                # --- 4a. Prepare Observations ---
                obs_to_predict = obs
                if self.running_average_obs:
                    mean_obs_batch = [np.mean(np.stack(deque), axis=0)
                                      for deque in running_avg_deques]
                    obs_to_predict = np.stack(mean_obs_batch)

                # --- 4b. Predict Action ---
                with torch.no_grad():
                    obs_to_predict = np.expand_dims(obs_to_predict, axis=-2)
                    if algo is None:
                        action = self.eval_env.action_space.sample()
                    else:
                        action, hidden_state = algo.predict(obs_to_predict,
                                                            hidden_state=hidden_state,
                                                            deterministic=True)

                # --- 4c. Step Environment ---
                try:
                    action_np = action.cpu().numpy()
                except AttributeError:
                    action_np = action

                if action_np.ndim == 1:
                    action_np = np.expand_dims(action_np, axis=-1)

                next_obs, reward, terminated, truncated, info = self.eval_env.step(action_np)
                dones = terminated | truncated

                episode_rewards += reward

                # --- 4d. Handle Dones and State Updates ---
                for i, done in enumerate(dones):
                    if self.running_average_obs:
                        running_avg_deques[i].append(next_obs[i])

                    if done:
                        episode_finished = False
                        # --- Balanced Logic: Add to the correct dictionary list ---
                        if do_balanced_eval:
                            current_id = self.env_id_map[i]
                            # Only add if this ID still needs episodes
                            if len(rewards_per_id[current_id]) < self.target_episodes_per_id:
                                rewards_per_id[current_id].append(episode_rewards[i])
                                episode_finished = True

                        # --- Original Logic: Add to the main list ---
                        else:
                            all_episode_rewards.append(episode_rewards[i])
                            episode_finished = True

                        if episode_finished:
                            pbar.update(1)

                        episode_rewards[i] = 0

                        if isinstance(hidden_state, tuple):
                            hidden_state[0][:, i, :] = 0
                            hidden_state[1][:, i, :] = 0
                        elif hidden_state is not None:
                            hidden_state[:, i, :] = 0

                        if self.running_average_obs:
                            running_avg_deques[i].clear()
                            running_avg_deques[i].append(next_obs[i])

                obs = next_obs

        pbar.clear()

        # --- 5. Combine and Return Results ---
        if self.test_ids is not None:
            final_rewards = []
            for id in self.test_ids:
                # Ensures we only return exactly `target_episodes_per_id` for each
                final_rewards.extend(rewards_per_id[id][:self.target_episodes_per_id])
        else:
            final_rewards = all_episode_rewards[:self.total_target_episodes]

        if original_device is not None:
            algo.to(original_device)
            algo._device = original_device

        return final_rewards


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
    options = os.listdir("../logs_glucose/ppo_logs")
    options = [i for i in options if i.endswith('.zip')]
    question = [inquirer.List('option',
                              message=message,
                              choices=options)]
    answer = inquirer.prompt(question)
    return {answer['option']}
