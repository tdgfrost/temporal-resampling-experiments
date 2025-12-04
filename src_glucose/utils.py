import glob
from copy import deepcopy
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

from gym_wrappers import *
from gymnasium.vector import AsyncVectorEnv


class RecurrentReplayBufferEnv:
    def __init__(self, env, buffer_size: int = 100000, sequence_length: int = 64, burn_in_length: int = 20,
                 reward_mean: Optional[dict] = None, reward_std: Optional[dict] = None):
        self.observations = {i: deque(maxlen=buffer_size) for i in range(4)}
        self.actions = {i: deque(maxlen=buffer_size) for i in range(4)}
        self.rewards = {i: deque(maxlen=buffer_size) for i in range(4)}
        self.dones = {i: deque(maxlen=buffer_size) for i in range(4)}
        self.sample_bool = {i: deque(maxlen=buffer_size) for i in range(4)}
        self.visible_states = {i: deque(maxlen=buffer_size) for i in range(4)}
        self.time_remaining = {i: deque(maxlen=buffer_size) for i in range(4)}

        self.buffer_size = buffer_size
        self.env = env
        self._tensors_set = False
        self._device = None
        self.batch_size = 32
        self.decoy_interval = 0
        self.n_samples = 0
        self.segments = 0
        self.include_time_remaining = False
        self.dataset_IQR_return = None
        self.dataset_IQR_std = None
        self.dataset_IQR_n_episodes = None
        self.reward_mean = {i: None for i in range(4)} if reward_mean is None else deepcopy(reward_mean)
        self.reward_std = {i: None for i in range(4)} if reward_std is None else deepcopy(reward_std)
        self.max_sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.sequence_info = {i: [] for i in range(4)}
        self.indices_map = {i: None for i in range(4)}
        self.padding_mask_map = {i: None for i in range(4)}
        self.train_mask_map = {i: None for i in range(4)}

    def __iter__(self):
        return iter(self.generate())

    def __len__(self):
        return self.segments

    def set_generate_params(self, device: str = 'cpu', batch_size: int = None, decoy_interval: int = None,
                            max_sequence_length: int = None, burn_in_length: int = None, epoch_fraction: float = 1.0,
                            include_time_remaining: bool = False):
        """
        Scans the buffer to identify all episodes and their lengths, preparing for sampling.
        """
        self.batch_size = batch_size if batch_size is not None else self.batch_size
        self.decoy_interval = decoy_interval if decoy_interval is not None else self.decoy_interval
        self.max_sequence_length = max_sequence_length if max_sequence_length is not None else self.max_sequence_length
        self.burn_in_length = burn_in_length if burn_in_length is not None else self.burn_in_length
        self.include_time_remaining = include_time_remaining if include_time_remaining is not None else self.include_time_remaining
        # Record the device for later
        self._device = device

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

        seq_info = self.sequence_info[self.decoy_interval]
        if len(seq_info) == 0:
            self.n_samples = 0
            self.segments = 0
            return

        self.n_samples = len(seq_info)

        # Calculate segments
        total_segments = self.n_samples // self.batch_size
        self.segments = int(total_segments * epoch_fraction) if total_segments > 0 else (1 if self.n_samples > 0 else 0)

        # Dimensions
        train_starts = seq_info[:, 0]  # Start of the training window
        actual_lens = seq_info[:, 1]  # Length of the training window
        ep_starts = seq_info[:, 2]  # Start of the episode (for burn-in clipping)

        max_fetch = self.burn_in_length + self.max_sequence_length

        # 1. Vectorized Index Grid
        # Shape: [N_samples, Max_Fetch_Len]
        # range_grid: [0, 1, 2, ... max_fetch-1]
        range_grid = np.arange(max_fetch)

        # Calculate the "Ideal" start index (Training Start - Burn In)
        ideal_fetch_starts = train_starts - self.burn_in_length

        # Calculate the "Actual" start index (Clamped at Episode Start)
        # This handles the case where burn-in goes "before" the episode began
        actual_fetch_starts = np.maximum(ep_starts, ideal_fetch_starts)

        # Create the Master Index Map: broadcast (N, 1) + (1, L)
        # This matrix contains the raw buffer indices for every timestep of every sample
        indices_map = actual_fetch_starts[:, None] + range_grid[None, :]

        # 2. Calculate Masks
        # How much burn-in did we actually get?
        actual_burn_in_lens = train_starts - actual_fetch_starts

        # Total valid length (Actual Burn In + Actual Training Length)
        total_valid_lens = actual_burn_in_lens + actual_lens

        # Padding Mask: False where index > total_valid_length
        # (1, L) < (N, 1)
        padding_mask_map = range_grid[None, :] < total_valid_lens[:, None]

        # Train Mask: True ONLY during the training phase (after burn-in)
        # Must be valid data AND index >= burn_in_len
        is_post_burnin = range_grid[None, :] >= actual_burn_in_lens[:, None]
        train_mask_map = padding_mask_map & is_post_burnin

        # 3. Convert to Tensors and Store on Device
        # By moving these to GPU now, fetching becomes purely indexing VRAM
        self.indices_map[self.decoy_interval] = torch.from_numpy(indices_map).long().to(device, non_blocking=True)

        # Masks need to be stored as [N, L, 1] for easy multiplication later
        self.padding_mask_map[self.decoy_interval] = torch.from_numpy(padding_mask_map).bool().unsqueeze(-1).to(device, non_blocking=True)
        self.train_mask_map[self.decoy_interval] = torch.from_numpy(train_mask_map).bool().unsqueeze(-1).to(device, non_blocking=True)

        # To avoid fetching indices that don't exist, set the padding mask 'false' indices to zero
        self.indices_map[self.decoy_interval] *= self.padding_mask_map[self.decoy_interval].squeeze(-1).long()

        # Ensure the raw data is on the device too (if not already)
        if not self._tensors_set:
            self.set_to_tensors(device=device)

    def generate(self):
        """
        Yields infinite batches of sequences for training.
        Generates a large pool of indices on the GPU to minimize CPU-GPU communication.
        """
        assert self.segments > 0, "Not enough episodes to generate a batch. Call set_generate_params()."

        # Configuration: How many batches to prepare in advance
        # 1000 batches is usually a sweet spot between memory usage and compute frequency
        batches_to_preload = 1000
        preload_count = self.batch_size * batches_to_preload

        while True:
            # 1. Generate a massive tensor of random indices directly on the device
            # This avoids Python for-loops and Host-to-Device transfers completely
            indices_buffer = torch.randint(
                low=0,
                high=self.n_samples,
                size=(preload_count,),
                device=self._device,
                dtype=torch.long
            )

            # 2. Iterate through the buffer in chunks
            for i in range(0, preload_count, self.batch_size):
                # Slicing a tensor is a "view" operation (zero overhead)
                batch_indices = indices_buffer[i: i + self.batch_size]

                # Pass the GPU tensor directly to fetch_transition_batch
                yield self.fetch_transition_batch(batch_indices, decoy_interval=self.decoy_interval)

    def generate_initial_states(self, batch_size: int = 1024, burn_in_window: int = 1):
        """
        Generator that yields batches containing the first 'burn_in_window' states
        of every valid episode found in the buffer.

        Episodes shorter than 'burn_in_window' are excluded.

        Yields:
             Tuple matching fetch_transition_batch output.
             Observations will have shape [Batch, burn_in_window, Obs_Dim]
        """
        # Resolve device
        device = self._device if self._device else 'cpu'

        # 1. Isolate Dones to find episode starts
        dones_data = self.dones[self.decoy_interval]

        # Handle different data types (List, Deque, or Tensor) to get a CPU Numpy mask
        if isinstance(dones_data, torch.Tensor):
            dones_np = dones_data.cpu().numpy().flatten()
        else:
            dones_np = np.array(dones_data, dtype=bool).flatten()

        if len(dones_np) == 0:
            return

        # 2. Identify Start Indices and Episode Lengths
        done_idxs = np.where(dones_np)[0]

        # Starts are 0 and one step after every done (that isn't the last index)
        potential_starts = np.concatenate(([0], done_idxs + 1))

        # We need corresponding ends to calculate length
        # If the last index in buffer isn't a done, it's still an end of a segment
        if len(done_idxs) > 0 and done_idxs[-1] == len(dones_np) - 1:
            ends = done_idxs
        else:
            # Append the end of the buffer as a virtual end for the last episode
            ends = np.concatenate((done_idxs, [len(dones_np) - 1]))

        # Ensure starts and ends are aligned.
        # Potential starts might have one extra if the last done was the last frame.
        if len(potential_starts) > len(ends):
            potential_starts = potential_starts[:len(ends)]

        lengths = ends - potential_starts + 1

        # 3. Filter Episodes based on Burn-In Window
        valid_mask = lengths >= burn_in_window
        valid_starts = potential_starts[valid_mask]
        n_starts = len(valid_starts)

        if n_starts == 0:
            return

        # 4. Create batches
        indices = np.arange(n_starts)

        for start_i in range(0, n_starts, batch_size):
            batch_indices = indices[start_i: start_i + batch_size]
            # Get the actual buffer indices for the *start* of these episodes
            start_buffer_idxs = valid_starts[batch_indices]

            current_bs = len(start_buffer_idxs)

            # Create the 2D grid of indices: [Batch, Time]
            # Shape: (Batch, burn_in_window)
            time_offset = np.arange(burn_in_window)
            buffer_idxs_2d = start_buffer_idxs[:, None] + time_offset[None, :]

            # Flatten for fetching, then we will reshape
            flat_buffer_idxs = buffer_idxs_2d.flatten()

            # 5. Fetch and Format Data
            def _get_tensor(source_dict, flat_idxs, cast_float=True):
                data = source_dict[self.decoy_interval]

                # Slicing logic depending on container type
                if isinstance(data, (list, deque)):
                    batch_data = np.array([data[i] for i in flat_idxs])
                elif isinstance(data, torch.Tensor):
                    batch_data = data[flat_idxs].cpu().numpy()
                else:
                    batch_data = data[flat_idxs]

                # Reshape back to [Batch, Time, Dim]
                # Note: Raw data might be (N,) or (N, D)
                if batch_data.ndim > 1:
                    # (N*T, D) -> (N, T, D)
                    batch_data = batch_data.reshape(current_bs, burn_in_window, -1)
                else:
                    # (N*T,) -> (N, T, 1)
                    batch_data = batch_data.reshape(current_bs, burn_in_window, 1)

                t = torch.from_numpy(batch_data).to(device, non_blocking=True)
                return t.float() if cast_float else t

            # Fetch data using the grid
            obs = _get_tensor(self.observations, flat_buffer_idxs)
            action = _get_tensor(self.actions, flat_buffer_idxs)
            reward = _get_tensor(self.rewards, flat_buffer_idxs)

            # Next Obs logic: Shift the whole grid by 1
            # Clamp to buffer end to avoid crash, though masks handle validity
            flat_next_idxs = np.minimum(flat_buffer_idxs + 1, len(dones_np) - 1)
            next_obs = _get_tensor(self.observations, flat_next_idxs)
            next_action = _get_tensor(self.actions, flat_next_idxs)

            # (Optional) Include time remaining
            if self.include_time_remaining:
                time_remaining = _get_tensor(self.time_remaining, flat_buffer_idxs)
                obs = torch.cat([obs, time_remaining], dim=-1)

                next_time_remaining = _get_tensor(self.time_remaining, flat_next_idxs)
                next_obs = torch.cat([next_obs, next_time_remaining], dim=-1)

            # Dones/Visible
            done = _get_tensor(self.dones, flat_buffer_idxs, cast_float=False).bool()
            visible = _get_tensor(self.visible_states, flat_buffer_idxs, cast_float=False).bool()
            next_visible = _get_tensor(self.visible_states, flat_next_idxs, cast_float=False).bool()

            # 6. Construct Masks
            # Since we filtered for validity, the whole window is valid "Train" data
            # Padding is False (we guaranteed data exists), Train is True

            # Shape: [Batch, T, 1]
            padding_mask = torch.zeros((current_bs, burn_in_window, 1), dtype=torch.bool, device=device)

            # Next padding mask: The last step of the sequence has no valid next step in this context
            next_padding_mask = torch.zeros_like(padding_mask)

            # For the very last index of the buffer, next is invalid
            at_buffer_limit = torch.tensor(flat_buffer_idxs == (len(dones_np) - 1), device=device)
            at_buffer_limit = at_buffer_limit.reshape(current_bs, burn_in_window, 1)
            next_padding_mask = next_padding_mask | at_buffer_limit

            train_mask = torch.ones((current_bs, burn_in_window, 1), dtype=torch.bool, device=device)

            yield (obs, action, reward, done, next_obs, next_action,
                   visible, next_visible, padding_mask, next_padding_mask, train_mask)

    def generate_all_trajectories(self, batch_size: int = 1024, decoy_interval: int = None):
        """
        Generator that yields ALL valid sliding window sequences in the dataset
        sequentially (not shuffled), exactly once.
        """
        if decoy_interval is None:
            decoy_interval = self.decoy_interval

        # Ensure parameters are set
        if self.n_samples == 0:
            print("Warning: No samples found. Did you call set_generate_params()?")
            return

        # Create sequential indices [0, 1, 2, ... N] - final truncated batch is dropped
        n_samples = (self.n_samples // batch_size) * batch_size
        indices = np.arange(n_samples)

        # Iterate in chunks
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield self.fetch_transition_batch(batch_indices, decoy_interval=decoy_interval)

    def fetch_transition_batch(self, idxs: np.ndarray, decoy_interval: int = 0):
        """
        O(1) Fetching using pre-calculated GPU maps.
        """
        # 1. Convert requested sample indices to tensor
        # (Using torch.tensor is fast for small arrays like batch_size=32)
        batch_idxs = torch.as_tensor(idxs, device=self._device, dtype=torch.long)

        # 2. Retrieve the specific indices and masks for this batch from the Maps
        # These lookups are extremely fast (GPU indexing)
        # Shape: [Batch, Max_Fetch_Len]
        gather_indices = self.indices_map[decoy_interval][batch_idxs]

        # Shape: [Batch, Max_Fetch_Len, 1]
        padding_mask = self.padding_mask_map[decoy_interval][batch_idxs]
        train_mask = self.train_mask_map[decoy_interval][batch_idxs]

        # 3. Retrieve Next Indices
        # We simply shift the gather indices by 1.
        max_possible_idx = self.indices_map[decoy_interval].max()
        next_gather_indices = torch.clip(gather_indices + 1, max=max_possible_idx)

        # 4. Gather Data
        # We use standard PyTorch advanced indexing: Tensor[Tensor]
        # This is highly optimized on GPU.

        obs = self.observations[decoy_interval][gather_indices]
        next_obs = self.observations[decoy_interval][next_gather_indices]

        # Actions/Rewards/Dones usually need unsqueeze(-1) if stored as flat [N]
        # If they are stored as [N, 1], remove the unsqueeze.
        # Assuming they are stored as [N, dim] based on previous code.

        def _gather_helper(source):
            data = source[decoy_interval][gather_indices]
            if data.ndim == 2:  # If [Batch, Seq], add feature dim
                return data.unsqueeze(-1)
            return data

        action = _gather_helper(self.actions)
        reward = _gather_helper(self.rewards)
        done = _gather_helper(self.dones)
        visible = _gather_helper(self.visible_states)

        # (Optional) get time_remaining and concat to obs
        if self.include_time_remaining:
            time_remaining = self.time_remaining[decoy_interval][gather_indices]
            if time_remaining.ndim == 2:
                time_remaining = time_remaining.unsqueeze(-1)

            obs = torch.cat([obs, time_remaining], dim=-1)

            # Next Time Remaining
            next_time_remaining = self.time_remaining[decoy_interval][next_gather_indices]
            if next_time_remaining.ndim == 2:
                next_time_remaining = next_time_remaining.unsqueeze(-1)

            next_obs = torch.cat([next_obs, next_time_remaining], dim=-1)

        # Next Action
        next_action = self.actions[decoy_interval][next_gather_indices]
        if next_action.ndim == 2:
            next_action = next_action.unsqueeze(-1)

        # Next Visible
        # Or gather using next_gather_indices
        next_visible = self.visible_states[decoy_interval][next_gather_indices]
        if next_visible.ndim == 2:
            next_visible = next_visible.unsqueeze(-1)

        # 5. Create Next Padding Mask
        # Shift padding mask one step to the left
        next_padding_mask = torch.roll(padding_mask, shifts=-1, dims=1)
        # The last step of a sequence never has a valid 'next', set to False
        next_padding_mask[:, -1, :] = False

        return (obs, action, reward, done, next_obs, next_action, visible, next_visible, padding_mask, next_padding_mask,
                train_mask)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for key, value in [
            ('observations', self.observations),
            ('actions', self.actions),
            ('rewards', self.rewards),
            ('dones', self.dones),
            ('sample_bool', self.sample_bool),
            ('visible_states', self.visible_states),
            ('time_remaining', self.time_remaining),
        ]:
            save_dict = {i: list(value[i]) for i in range(4)}
            np.savez(os.path.join(path, f'{key}.npz'),
                     **{str(k): v for k, v in save_dict.items()})

        # Save dataset IQR return and scaling
        save_dict = {'dataset_iqr_avg': self.dataset_IQR_return, 'dataset_iqr_std': self.dataset_IQR_std,
                     'dataset_iqr_n_episodes': self.dataset_IQR_n_episodes}
        save_dict.update({f'reward_mean_{i}': self.reward_mean[i] for i in range(4)})
        save_dict.update({f'reward_std_{i}': self.reward_std[i] for i in range(4)})
        np.savez(os.path.join(path, 'rewards_scale.npz'), **save_dict)

        # Mark saving as complete
        with open(os.path.join(path, 'COMPLETE'), 'w') as f:
            f.close()

    def load(self, path: str, reduce_fraction: Optional[float] = None):
        for key, value in [
            ('observations', self.observations),
            ('actions', self.actions),
            ('rewards', self.rewards),
            ('dones', self.dones),
            ('sample_bool', self.sample_bool),
            ('visible_states', self.visible_states),
            ('time_remaining', self.time_remaining),
        ]:
            loaded = np.load(os.path.join(path, f'{key}.npz'), allow_pickle=True)
            for k in loaded.files:
                value[int(k)] = loaded[k]

        # Load dataset IQR return and scaling
        loaded_scale = np.load(os.path.join(path, 'rewards_scale.npz'), allow_pickle=True)
        self.dataset_IQR_return = float(loaded_scale['dataset_iqr_avg'])
        self.dataset_IQR_std = float(loaded_scale['dataset_iqr_std'])
        self.dataset_IQR_n_episodes = int(loaded_scale['dataset_iqr_n_episodes'])

        for i in range(4):
            self.reward_mean[i] = float(loaded_scale[f'reward_mean_{i}'])
            self.reward_std[i] = float(loaded_scale[f'reward_std_{i}'])

        # Tweak to shorten dataset for faster evaluation if required
        if reduce_fraction is not None:
            assert 0 < reduce_fraction < 1.0, "reduce_fraction must be between 0 and 1"
            for i in range(4):
                done_indices = np.where(self.dones[i])[0]
                n_episodes = int(done_indices.shape[0] * reduce_fraction)
                if i == 0:
                    print(f"Trimming {done_indices.shape[0]} to {n_episodes} episodes for {path}...")
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
            'visible_state': [True],
            'steps_remaining': [1.0],
        }

    def fill_buffer(self, model, n_frames: int = 1_000, seed: int = None, save_path: str = "./replay_buffer",
                    chunk_size: int = 20000):
        """
        Generates data and saves to temporary chunks to avoid memory overflow.
        At the end, loads chunks, normalizes, and saves the final dataset.
        """
        def create_new_seed(base_seed, offset):
            return np.random.SeedSequence([base_seed, offset]).generate_state(1)[0].item()
        # Create a temporary directory for chunks
        temp_dir = os.path.join(save_path, "temp_chunks")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # Initialize temporary in-memory storage for the current chunk
        # We use lists here because they are faster for appending than deques for temporary storage
        chunk_storage = self._init_storage_dict()

        with tqdm(total=n_frames, desc="Progress", mininterval=2.0) as pbar:
            frame_count = 0
            chunk_idx = 0
            if seed is None:
                seed = 123
            current_seed = create_new_seed(seed, frame_count)

            obs, info, ep_buffer = self.reset(seed=current_seed)
            total_rewards = []
            total_lengths = []
            max_episode_steps = MAX_STEPS

            while frame_count < n_frames:
                model.set_random_seed(current_seed)
                done = False
                lstm_states = model.ac_network.init_hidden_state(batch_size=1)
                total_reward = 0
                current_ep_steps = 0

                while not done:
                    action, lstm_states = model.predict(np.expand_dims(obs, 0), hidden_state=lstm_states,
                                                        deterministic=False)
                    obs, reward, term, trunc, info = self.env.step(action)

                    # Get the real action delivered to the patient
                    real_action = obs[1] * INSULIN_SCALE

                    done = term or trunc
                    total_reward += reward
                    current_ep_steps += 1
                    steps_remaining = (max_episode_steps - current_ep_steps) / max_episode_steps

                    self.update_episode_buffer(obs, real_action, reward, term, trunc, info, ep_buffer, steps_remaining)

                    if done:
                        # Update our frame count seed
                        current_seed = create_new_seed(seed, frame_count)
                        # Reset our buffer and environment
                        ep_buffer['all_obs'] = ep_buffer['all_obs'][:-1]
                        obs, info = self.env.reset(seed=current_seed)
                        current_ep_steps = 0

                    pbar.update(1)

                    frame_count += 1

                # Keep track of total rewards for stats (this is small, keep in RAM)
                total_rewards.append(total_reward)
                total_lengths.append(len(ep_buffer['all_done']))

                # Update progress bar with stats (every 500 episodes)
                if (len(total_rewards) + 1) % 500 == 0:
                    # This is a slow operation, so we do it infrequently
                    iqm = np.mean(trimboth(np.array(total_rewards), 0.25))

                    pbar_dict = {
                        'avg_ep_r': f"{np.mean(total_rewards):.2f}",
                        'avg_ep_IQM': f"{iqm:.2f}",
                        'refresh': False
                    }
                    pbar.set_postfix(**pbar_dict)

                # Add ep_buffer to our TEMPORARY chunk storage
                self.update_permanent_buffer(ep_buffer, storage=chunk_storage)

                # Reset ep_buffer
                ep_buffer = self._reset_ep_buffer(obs)

                # --- CHECK IF CHUNK IS FULL ---
                # Check size of observations[0]
                if len(chunk_storage['observations'][0]) >= chunk_size:
                    self._save_chunk(chunk_storage, temp_dir, chunk_idx)
                    chunk_idx += 1
                    # Reset storage
                    chunk_storage = self._init_storage_dict()

            # Save any remaining data in the final partial chunk
            if len(chunk_storage['observations'][0]) > 0:
                self._save_chunk(chunk_storage, temp_dir, chunk_idx)

        print("Data collection complete. Merging chunks and normalizing...")

        # --- FINALIZE ---
        # 1. Calculate IQR stats
        total_rewards = np.array(total_rewards)
        IQR = trimboth(total_rewards, proportiontocut=0.25)
        self.dataset_IQR_return = IQR.mean()
        self.dataset_IQR_std = IQR.std()
        self.dataset_IQR_n_episodes = len(IQR)

        # 2. Load all chunks, merge, normalize, and save final
        self._finalize_dataset(temp_dir, save_path)

        # Cleanup temp dir
        shutil.rmtree(temp_dir)
        print(f"Dataset saved to {save_path}")

    @staticmethod
    def _init_storage_dict():
        """Creates an empty dictionary structure mimicking self.observations etc."""
        return {
            'observations': {i: [] for i in range(4)},
            'actions': {i: [] for i in range(4)},
            'rewards': {i: [] for i in range(4)},
            'dones': {i: [] for i in range(4)},
            'sample_bool': {i: [] for i in range(4)},
            'visible_states': {i: [] for i in range(4)},
            'time_remaining': {i: [] for i in range(4)}
        }

    @staticmethod
    def _save_chunk(storage, temp_dir, chunk_idx):
        """Saves the current dictionary of lists to an NPZ file."""
        save_dict = {}
        for key in ['observations', 'actions', 'rewards', 'dones', 'sample_bool', 'visible_states', 'time_remaining']:
            for i in range(4):
                # Convert list to array for saving
                save_dict[f"{key}_{i}"] = np.array(storage[key][i])

        filename = os.path.join(temp_dir, f"chunk_{chunk_idx}.npz")
        np.savez_compressed(filename, **save_dict)

    def _finalize_dataset(self, temp_dir, save_path):
        """Loads all chunks, merges them, normalizes rewards, and saves permanently."""
        chunk_files = sorted(glob.glob(os.path.join(temp_dir, "chunk_*.npz")),
                             key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

        if not chunk_files:
            print("No chunks found!")
            return

        # We will load keys one by one to save memory, merge, and put into self
        keys = ['observations', 'actions', 'rewards', 'dones', 'sample_bool', 'visible_states']

        # Temporary holding for merged arrays
        merged_data = {k: {i: [] for i in range(4)} for k in keys}

        print("Loading chunks...")
        for cf in chunk_files:
            data = np.load(cf, allow_pickle=True)
            for k in keys:
                for i in range(4):
                    arr = data[f"{k}_{i}"]
                    if len(arr) > 0:
                        merged_data[k][i].append(arr)

        print("Concatenating and Normalizing...")
        for i in range(4):
            # Concatenate arrays
            # For observations, we might want to add the "final zero obs" here if needed
            # The original code added a zero obs at the very end.

            # --- REWARDS (Normalize) ---
            r_list = merged_data['rewards'][i]
            if r_list:
                full_rewards = np.concatenate(r_list, axis=0)

                r_mean = self.reward_mean[i] or full_rewards.mean()
                r_std = self.reward_std[i] or full_rewards.std()

                norm_rewards = (full_rewards - r_mean) / (r_std + 1e-8)

                self.rewards[i] = deque(norm_rewards, maxlen=self.buffer_size)
                self.reward_mean[i] = r_mean
                self.reward_std[i] = r_std
            else:
                self.rewards[i] = deque([], maxlen=self.buffer_size)

            # --- OTHERS (Just load) ---
            for k in ['observations', 'actions', 'dones', 'sample_bool', 'visible_states']:
                arr_list = merged_data[k][i]
                if arr_list:
                    full_arr = np.concatenate(arr_list, axis=0)

                    # Special handling for observations padding if required by original logic
                    # (Original code added one zero-obs at the end of fill_buffer)
                    if k == 'observations' and i < 3:
                        # Only add the padding to the very end of the concatenated array
                        # if strict compatibility with original "final obs" logic is needed
                        zero_pad = np.zeros_like(full_arr[0:1])
                        full_arr = np.concatenate([full_arr, zero_pad], axis=0)

                    self.__dict__[k][i] = deque(full_arr, maxlen=self.buffer_size)
                else:
                    self.__dict__[k][i] = deque([], maxlen=self.buffer_size)

        # Now save using the existing save method
        self.save(save_path)

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
        self.time_remaining[i] = _set_device(self.time_remaining[i]).float()
        self.sequence_info[i] = _set_device(self.sequence_info[i]).long()

        self._tensors_set = True
        self._device = device

    @staticmethod
    def update_episode_buffer(obs, action: Union[int, np.ndarray], reward: float, term: bool, trunc: bool, info: dict,
                              ep_buffer: dict, steps_remaining: int):
        ep_buffer['all_obs'] += [obs]
        ep_buffer['all_action'] += [action]
        ep_buffer['all_reward'] += [reward]
        ep_buffer['all_term'] += [term]
        ep_buffer['all_trunc'] += [trunc]
        ep_buffer['all_done'] += [term or trunc]
        ep_buffer['visible_state'] += [info['steps_until_action_available'] == 0]
        ep_buffer['steps_remaining'] += [steps_remaining]

    @staticmethod
    def _get_aggregated_episode(ep_buffer, window_size):
        # Helper function for update_permanent_buffer
        obs_data = np.array(ep_buffer['all_obs'])
        action_data = np.array(ep_buffer['all_action'])
        done_data = np.array(ep_buffer['all_done'])
        reward_data = np.array(ep_buffer['all_reward'])

        # Generate our aggregated datapoints for decoy_interval = 2 and 3
        # e.g., for window_size = 3, we use the following aggregates:
        # for i in [3, 6, 9, ...]
        # [t0,  t1,  t2,  t3,  t4,  t5,  t6,  ...]
        # obs:  |----------|    |---------|    |----   (t0-t2, t3-t5, etc)
        # act:             |---------|    |-------     (t2-t4, t5-t7, etc)
        # rew:             |---------|    |-------     (t2-t4, t5-t7, etc)
        # (reward calculated from next obs)

        # Aggregate obs
        obs_indices = range(window_size, len(obs_data) + 1, window_size)
        obs_splits = np.array_split(obs_data, obs_indices, axis=0)[:-1]
        obs = [np.mean(chunk, axis=0) for chunk in obs_splits if chunk.size > 0]

        # Aggregate actions / dones / rewards
        slice_indices = range(window_size - 1, len(obs_data), window_size)
        action_splits = np.array_split(action_data, slice_indices, axis=0)[1:]
        done_splits = np.array_split(done_data, slice_indices, axis=0)[1:]
        reward_splits = np.array_split(reward_data, slice_indices, axis=0)[1:]

        actions = [np.mean(chunk, axis=0) for chunk in action_splits if chunk.size > 0]
        dones = [np.any(chunk) for chunk in done_splits if chunk.size > 0]
        rewards = [np.sum(chunk) for chunk in reward_splits if chunk.size > 0]

        return obs, actions, rewards, dones

    def update_permanent_buffer(self, ep_buffer: dict, storage=None):
        """
        Added `storage` argument.
        If storage is None, behaves like original (writes to self).
        If storage is provided (dict), writes to that dict (used for chunking).
        """
        # Select target container
        if storage is None:
            target_obs = self.observations
            target_act = self.actions
            target_rew = self.rewards
            target_don = self.dones
            target_smp = self.sample_bool
            target_vis = self.visible_states
            target_time = self.time_remaining
        else:
            target_obs = storage['observations']
            target_act = storage['actions']
            target_rew = storage['rewards']
            target_don = storage['dones']
            target_smp = storage['sample_bool']
            target_vis = storage['visible_states']
            target_time = storage['time_remaining']

        visible_idxs = ep_buffer['visible_state']
        for i in range(2):
            target_obs[i] += ep_buffer['all_obs']
            target_act[i] += ep_buffer['all_action']
            target_rew[i] += ep_buffer['all_reward']
            target_don[i] += ep_buffer['all_done']
            target_smp[i] += [True for _ in range(len(ep_buffer['all_done']))]
            target_time[i] += ep_buffer['steps_remaining']

        # Update visible states
        target_vis[0] += visible_idxs
        target_vis[1] += [True for _ in range(len(ep_buffer['all_done']))]

        # Calculate Index 2 (4 Hour, using AGGREGATE_WINDOW_SIZE)
        # Calculate Index 3 (2 Hour, using AGGREGATE_WINDOW_SIZE // 2)

        assert AGGREGATE_WINDOW_SIZE * SAMPLE_TIME // 60 == 4, \
            ("Current code assumes aggregation window size corresponds to 4 hours. "
             "Is SAMPLE_TIME still 10 minutes? If so, the following code needs to be checked.")

        windows = {
            2: AGGREGATE_WINDOW_SIZE,  # 4 hours
            3: max(1, AGGREGATE_WINDOW_SIZE // 2)  # 2 hours
        }

        for idx, win_size in windows.items():
            obs, actions, rewards, dones = self._get_aggregated_episode(ep_buffer, win_size)

            # Make sure this is a valid minimum-length episode we are saving
            if dones and dones[-1]:
                target_obs[idx] += obs
                target_act[idx] += actions
                target_rew[idx] += rewards
                target_don[idx] += dones
                target_smp[idx] += [True for _ in range(len(dones))]
                target_vis[idx] += [True for _ in range(len(dones))]

                # For steps remaining, this should be based on the NEW max length of the aggregated episode
                raw_max = MAX_STEPS
                agg_max = int(raw_max / win_size)

                agg_time_remaining = []
                for t in range(len(dones)):
                    val = max(0, agg_max - t) / agg_max
                    agg_time_remaining.append(val)

                target_time[idx] += agg_time_remaining

                raise Exception("Check the above code works as intended")


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

                try:
                    action_np = action_np.reshape(-1, 1)
                except ValueError:
                    raise ValueError("Action could not be reshaped to (-1, 1). Check action dimensions.")

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


class FQEEvaluator:
    def __init__(
            self,
            dataset,
            batch_size: int = 1024,
            return_loss: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.return_loss = return_loss

    def __call__(self, algo, seed=None):
        """
        Goal should be to iterate through the whole dataset once (first state only), calculating the FQE estimate.
        """
        if self.return_loss:
            losses = []
            # Iterate through the entire dataset once
            with tqdm(total=self.dataset.segments, desc="Calculating validation loss...", mininterval=2.0,
                      leave=False) as pbar:
                for batch in self.dataset.generate_all_trajectories(batch_size=self.batch_size):
                    loss = algo.get_validation_loss(*batch)
                    # Set the loss to be negative for maximisation, consistent with maximising scores for early stopping
                    losses.append(-1 * loss)
                    pbar.update(1)

            return np.array(losses)

        # --- Initial State Evaluation (with 4-hour burn-in) ---

        # 1. Determine Burn-in Window based on Decoy Interval
        # Goal: Treat S_0 as the state after 4 hours of behavior policy.
        decoy_interval = self.dataset.decoy_interval
        assert AGGREGATE_WINDOW_SIZE * SAMPLE_TIME // 60 == 4, \
            ("Current code assumes aggregation window size corresponds to 4 hours. "
             "Is SAMPLE_TIME still 10 minutes? If so, the following code needs to be checked carefully.")

        if decoy_interval == 2:
            # 1 step = 4 hours.
            burn_in_steps = 1
        elif decoy_interval == 3:
            # 1 step = 2 hours.
            burn_in_steps = 2
        else:
            # Raw data (Interval 0 or 1).
            # AGGREGATE_WINDOW_SIZE corresponds to 4 hours in raw steps.
            burn_in_steps = AGGREGATE_WINDOW_SIZE

        # 2. Get Scaling Data
        mu = self.dataset.reward_mean[decoy_interval]
        sigma = self.dataset.reward_std[decoy_interval]

        dones_data = self.dataset.dones[decoy_interval]
        if isinstance(dones_data, torch.Tensor):
            dones_np = dones_data.cpu().numpy()
        else:
            dones_np = np.array(dones_data, dtype=bool)

        n_episodes = np.sum(dones_np)
        total_steps = len(dones_np)
        mean_ep_length = total_steps / n_episodes if n_episodes > 0 else 0

        # 3. Iterate over valid episodes
        all_fqe_preds = []

        # We pass the calculated burn_in_steps here
        gen = self.dataset.generate_initial_states(batch_size=self.batch_size, burn_in_window=burn_in_steps)

        for batch in gen:
            (obs, acts, _, _, _, _, _, _, _, _, _) = batch

            # Remove steps_remaining for target model
            target_obs = obs[..., :-1]

            # Obs is shape [Batch, burn_in_steps, Dim]

            with torch.no_grad():
                # Predict sequence of actions
                # algo.target_model.predict returns actions for the whole sequence
                acts_preds, _ = algo.target_model.predict(target_obs, deterministic=True, action_as_tensor=True)

                if acts_preds is None:
                    acts_preds = acts

                # Get Q-values for the whole sequence
                # Shape: [Batch, burn_in_steps, 1]
                fqe_preds_seq = algo.get_value_estimate(obs, acts_preds)

                # We only want the value estimate at the END of the burn-in period (i.e., after 4 hours)
                # Take the last timestep
                fqe_preds = fqe_preds_seq[:, -1, :]

            all_fqe_preds.append(fqe_preds.squeeze())

        if len(all_fqe_preds) > 0:
            all_fqe_preds = torch.cat(all_fqe_preds).cpu().numpy()  # [N]
        else:
            print("Warning: No episodes found with sufficient length for 4-hour burn-in.")
            return np.array([0.0])

        # Rescale q_preds back to original reward scale
        raw_fqe_preds = all_fqe_preds * sigma + (mu * mean_ep_length)

        return raw_fqe_preds


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
    options = [i for i in options if i.endswith('.pt')]
    question = [inquirer.List('option',
                              message=message,
                              choices=options)]
    answer = inquirer.prompt(question)
    return {answer['option']}


def choose_offline_agent(model_type: str, decoy_interval: int):
    # Use inquirer to let the user select a folder
    message = "Please select offline agent using UP/DOWN/ENTER."
    options = os.listdir(f"../logs_glucose/iql_models/decoy_interval_{decoy_interval}/{model_type}")
    options = [i for i in options if i.endswith('.pt')]
    question = [inquirer.List('option',
                              message=message,
                              choices=options)]
    answer = inquirer.prompt(question)
    return {answer['option']}


def load_buffer_datasets(fill_if_absent: bool = False, ppo_agent=None, dataset_size: int = 10_000_000,
                         reduce_fraction: Optional[Dict[str, float]] = None):
    assert not fill_if_absent or ppo_agent is not None, \
        "PPO agent must be provided if not filling buffers."

    def create_empty_dataset(patient_ids: List[int], *args, **kwargs):
        return RecurrentReplayBufferEnv(
            make_glucose_env(patient_ids=patient_ids),
            buffer_size=dataset_size * 2,
            *args, **kwargs
        )

    datasets = dict()
    r_mean, r_std = None, None
    for key, patient_ids, n_frames in [
        ('train', TRAIN_IDS, dataset_size),
        ('val', VAL_IDS, dataset_size // 3),
        ('test', TEST_IDS, dataset_size // 3)
    ]:
        # Create empty dataset (+/- train dataset reward scaling)
        dataset = create_empty_dataset(patient_ids=patient_ids, reward_mean=r_mean, reward_std=r_std)

        if not os.path.exists(f'./replay_buffer_{key}/COMPLETE'):
            if fill_if_absent:
                print(f"n=== Generating replay buffer for {key} dataset ===\n")
                # Load PPO agent and fill buffer
                dataset.fill_buffer(model=ppo_agent, n_frames=dataset_size, save_path=f'./replay_buffer_{key}')
            else:
                raise FileNotFoundError(f"Replay buffer for {key} dataset not found at "
                                        f"'./replay_buffer_{key}/COMPLETE'. "
                                        f"Please run offline training to generate the datasets.")

        else:
            if isinstance(reduce_fraction, dict):
                current_reduce_fraction = reduce_fraction.get(key, None)
            else:
                current_reduce_fraction = reduce_fraction
            dataset.load(f'./replay_buffer_{key}', reduce_fraction=current_reduce_fraction)

        if key == 'train':
            r_mean, r_std = dataset.reward_mean, dataset.reward_std
        datasets[key] = dataset

    return datasets
