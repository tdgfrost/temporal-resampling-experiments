from typing import Union
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


class ReplayBufferEnv:
    def __init__(self, env, buffer_size: int = 100000):
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
        self.max_rewards_scale = None
        self.min_rewards_scale = None
        self.dataset_avg = None
        self.dataset_std = None

    def __iter__(self):
        return iter(self.generate())

    def __len__(self):
        return self.n_samples

    def set_generate_params(self, batch_size: int = None, decoy_interval: int = None):
        self.batch_size = batch_size if batch_size is not None else self.batch_size
        self.decoy_interval = decoy_interval if decoy_interval is not None else self.decoy_interval
        self.n_samples = sum(self.sample_bool[self.decoy_interval])
        self.segments = self.n_samples // self.batch_size if self.n_samples > 0 else 0

        assert self.n_samples >= self.batch_size, "Not enough samples in the replay buffer"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.decoy_interval in [0, 1, 2], "decoy_interval must be 0, 1, or 2"

    def generate(self):
        assert self.n_samples > 0, "Replay buffer is empty"
        # The base class samples from a contiguous range.
        valid_indices = np.where(self.sample_bool[self.decoy_interval])[0]
        # We must subtract 1 because fetch_transition_batch accesses idxs+1
        valid_indices = valid_indices[valid_indices < len(self.observations[self.decoy_interval]) - 1]

        self.n_samples = len(valid_indices)
        self.segments = self.n_samples // self.batch_size
        assert self.segments > 0, f"Not enough valid samples to form a batch. Have {self.n_samples}, need {self.batch_size}"

        idxs_np = np.random.choice(valid_indices, size=(self.segments, self.batch_size), replace=False)
        idxs = torch.tensor(idxs_np, dtype=torch.long, device=self._device)

        for idx in idxs:
            yield self.fetch_transition_batch(idxs=idx, decoy_interval=self.decoy_interval)

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

        # Also save min/max rewards scale and dataset avg
        np.savez(os.path.join(path, 'rewards_scale.npz'),
                 ** {'min': self.min_rewards_scale, 'max': self.max_rewards_scale,
                     'dataset_avg': self.dataset_avg, 'dataset_std': self.dataset_std})

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

        # Load min/max rewards scale
        loaded_scale = np.load(os.path.join(path, 'rewards_scale.npz'), allow_pickle=True)
        self.min_rewards_scale = float(loaded_scale['min'])
        self.max_rewards_scale = float(loaded_scale['max'])
        self.dataset_avg = float(loaded_scale['dataset_avg'])
        self.dataset_std = float(loaded_scale['dataset_std'])

    def reset(self, seed: int = None):
        obs, info = self.env.reset(seed=seed)
        ep_buffer = self._reset_ep_buffer(obs, info)
        return obs, info, ep_buffer

    @staticmethod
    def _reset_ep_buffer(obs, info):
        return {
            'all_obs': [obs],
            'all_action': [],
            'all_reward': [],
            'all_done': [],
            'visible_state': [obs[0] == 0]
        }

    def fill_buffer(self, model, n_frames: int = 1_000, seed: int = None, with_random: bool = True, rand_p: float = 0.05):
        with tqdm(total=n_frames, desc="Progress", mininterval=2.0) as pbar:
            frame_count = 0
            if seed is None:
                seed = 123

            obs, info, ep_buffer = self.reset(seed=seed)
            model.set_random_seed(seed)
            lstm_states = None
            total_rewards = []

            while frame_count < n_frames:
                done = False
                episode_starts = np.ones((1,), dtype=bool)
                total_reward = 0
                while not done:
                    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
                    # if with_random and np.random.random() < rand_p:
                        # action = self.env.action_space.sample()
                        # action = np.random.uniform(low=0, high=2, size=1).astype(np.float32)
                    obs, reward, term, trunc, info = self.env.step(action)
                    # Get the real action delivered
                    real_action = obs[3] * 30
                    done = term or trunc
                    episode_starts[0] = done
                    total_reward += reward

                    self.update_episode_buffer(obs, real_action, reward, done, info, ep_buffer)

                    if done:
                        ep_buffer['all_obs'] = ep_buffer['all_obs'][:-1]
                        obs, info = self.env.reset(seed=seed + frame_count)

                    pbar.update(1)
                    frame_count += 1
                    model.set_random_seed(seed + frame_count)

                # Add ep_buffer to permanent buffer
                self.update_permanent_buffer(ep_buffer)
                # Reset ep_buffer and add 'obs' to it
                ep_buffer = self._reset_ep_buffer(obs, info)

                # Keep track of total rewards for stats
                total_rewards.append(total_reward)

            # Add a garbage all-zeros "final obs"
            for i in range(3):
                self.observations[i] += [np.zeros_like(self.observations[0][0])]

            # Normalise rewards from 0 to 1
            min_r, max_r = min(total_rewards), 200 # max(total_rewards)
            for i in [0, 1, 2]:
                rewards = np.array(self.rewards[i])
                if max_r > min_r:
                    norm_rewards = (rewards - min_r) / (max_r - min_r)
                else:
                    norm_rewards = rewards - min_r  # All zeros
                self.rewards[i] = deque(norm_rewards.tolist(), maxlen=self.buffer_size)
                self.min_rewards_scale = min_r
                self.max_rewards_scale = max_r

            norm_total_rewards = (np.array(total_rewards) - min_r) / (max_r - min_r)
            self.dataset_avg = norm_total_rewards.mean()
            self.dataset_std = norm_total_rewards.std()

    def set_to_tensors(self, device: str = 'cpu'):
        if self._tensors_set and self._device == device:
            return

        for i in range(3):
            self.observations[i] = torch.from_numpy(np.array(self.observations[i])).float().to(device)
            self.actions[i] = torch.from_numpy(np.array(self.actions[i])).float().to(device)
            self.rewards[i] = torch.from_numpy(np.array(self.rewards[i])).float().to(device)
            self.dones[i] = torch.from_numpy(np.array(self.dones[i])).bool().to(device)
            self.visible_states[i] = torch.from_numpy(np.array(self.visible_states[i])).bool().to(device)

        self._tensors_set = True
        self._device = device

    @staticmethod
    def update_episode_buffer(obs, action: Union[int, np.ndarray], reward: float, done: bool, info: dict, ep_buffer: dict):
        ep_buffer['all_obs'] += [obs]
        ep_buffer['all_action'] += [action]
        ep_buffer['all_reward'] += [reward]
        ep_buffer['all_done'] += [done]
        ep_buffer['visible_state'] += [obs[0] == 0]

    def update_permanent_buffer(self, ep_buffer: dict):
        visible_idxs = ep_buffer['visible_state']
        for i in range(2):
            for arr, key in [
                (self.observations, 'obs'), (self.actions, 'action'), (self.rewards, 'reward'), (self.dones, 'done')
            ]:
                # if i == 1 and key == 'obs':
                if key == 'obs':
                    ep_buffer[f'all_{key}'] = np.stack(ep_buffer[f'all_{key}'])
                    ep_buffer[f'all_{key}'][:, :2] = 0  # Set the flags to 0 for the non-decoy buffer
                    ep_buffer[f'all_{key}'] = ep_buffer[f'all_{key}'].tolist()
                arr[i] += ep_buffer[f'all_{key}']

            self.sample_bool[i] += [True for _ in range(len(ep_buffer[f'all_done']))]

        # Update visible states
        self.visible_states[0] += visible_idxs
        self.visible_states[1] += [True for _ in range(len(ep_buffer[f'all_done']))]

        # Update the decoy actions (every second step)
        # - For basal insulin, this involves taking the average of the two actions
        actions = [
            np.mean(ep_buffer['all_action'][idx: idx + 12])
            for idx in range(12, len(ep_buffer['all_action']), 12)
        ]

        rewards, dones = [
            [np.sum(ep_buffer[f'all_{key}'][idx: idx + 12]) for idx in range(12, len(ep_buffer['all_reward']), 12)]
            for key in ['reward', 'done']
        ]  # Note to self <- should this be recalculated from scratch using the average blood glucose?

        # For obs, take the mean observation of the two steps
        # this is from t-11 to t0 (inclusive)
        obs = np.stack([
            np.mean(ep_buffer['all_obs'][idx-12 + 1: idx + 1], 0)
            for idx in range(12, len(ep_buffer['all_obs']), 12)
        ])
        # Change the steps_remaining flag to 0
        obs[:, :2] = 0
        obs = obs.tolist()

        # if not dones[-1]:
            # rewards[-1] = ep_buffer['reward'][-1]
            # dones[-1] = ep_buffer['done'][-1]
        assert dones[-1], "Last done flag should be True"
        self.observations[2] += obs
        self.actions[2] += actions
        self.rewards[2] += rewards
        self.dones[2] += dones
        self.sample_bool[2] += [True for _ in range(len(dones))]
        self.visible_states[2] += [True for _ in range(len(dones))]

    def sample_transition_batch(self, batch_size: int = 32, decoy_interval: int = 0):
        idxs = torch.randint(0, len(self.observations[decoy_interval]) - 1, (batch_size,), device=self._device)
        return self.fetch_transition_batch(idxs, decoy_interval)

    def fetch_transition_batch(self, idxs: torch.Tensor, decoy_interval: int = 0):
        assert self._tensors_set, "Replay buffer must be set to tensors first using .set_to_tensors(model)"
        obs_batch = self.observations[decoy_interval][idxs]
        next_obs_batch = self.observations[decoy_interval][idxs + 1]
        action_batch = self.actions[decoy_interval][idxs].unsqueeze(-1)
        reward_batch = self.rewards[decoy_interval][idxs].unsqueeze(-1)
        done_batch = self.dones[decoy_interval][idxs].unsqueeze(-1)

        flags = self._extract_flags(obs_batch)
        next_flags = self._extract_flags(next_obs_batch)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, flags, next_flags

    @staticmethod
    def _extract_flags(obs):
        return obs[..., 0].unsqueeze(-1).long()


class RecurrentReplayBufferEnv(ReplayBufferEnv):  # Inherits from your original class
    def __init__(self, env, buffer_size: int = 100000, sequence_length: int = 64):
        super().__init__(env, buffer_size)
        self.max_sequence_length = sequence_length
        # This will store the valid start indices for each decoy interval
        self.sequence_info = {i: [] for i in range(3)}

    def set_generate_params(self, device: str = 'cpu', batch_size: int = None, decoy_interval: int = None, max_sequence_length: int = None):
        """
        Scans the buffer to identify all episodes and their lengths, preparing for sampling.
        """
        self.batch_size = batch_size if batch_size is not None else self.batch_size
        self.decoy_interval = decoy_interval if decoy_interval is not None else self.decoy_interval
        self.max_sequence_length = max_sequence_length if max_sequence_length is not None else self.max_sequence_length

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
                    # Short episode: Add one sequence to be padded.
                    self.sequence_info[self.decoy_interval].append((start_idx, length))
                else:
                    # Long episode: Use sliding window.
                    last_possible_start = start_idx + length - self.max_sequence_length
                    for seq_start in range(start_idx, last_possible_start + 1):
                        self.sequence_info[self.decoy_interval].append((seq_start, self.max_sequence_length))

            # The next episode will start after the last 'done'
            last_ep_start_idx = done_indices[-1] + 1

        # --- ADDED: Handle the final, possibly unterminated episode ---
        final_buffer_len = len(dones_np)
        if last_ep_start_idx < final_buffer_len:
            final_ep_len = final_buffer_len - last_ep_start_idx

            if final_ep_len <= self.max_sequence_length:
                # The final short episode.
                self.sequence_info[self.decoy_interval].append((last_ep_start_idx, final_ep_len))
            else:
                # The final long episode.
                last_possible_start = last_ep_start_idx + final_ep_len - self.max_sequence_length
                for seq_start in range(last_ep_start_idx, last_possible_start + 1):
                    self.sequence_info[self.decoy_interval].append((seq_start, self.max_sequence_length))

        # n_samples is now the number of sequences in the buffer
        self.n_samples = len(self.sequence_info[self.decoy_interval])
        self.segments = self.n_samples // self.batch_size if self.n_samples > 0 else 0

        if self.segments == 0:
            print(f"Warning: Not enough episodes for sampling. "
                  f"Found {self.n_samples} episodes but batch size is {self.batch_size}.")

        self.set_to_tensors(device)

    def set_to_tensors(self, device: str = 'cpu'):
        def _set_device(some_arr):
            if isinstance(some_arr, torch.Tensor):
                return some_arr.to(device)
            return torch.from_numpy(np.array(some_arr)).to(device)

        i = self.decoy_interval

        self.observations[i] = _set_device(self.observations[i]).float()
        self.actions[i] = _set_device(self.actions[i]).float()
        self.rewards[i] = _set_device(self.rewards[i]).float()
        self.dones[i] = _set_device(self.dones[i]).bool()
        self.visible_states[i] = _set_device(self.visible_states[i]).bool()
        self.sequence_info[i] = _set_device(self.sequence_info[i]).long()

        self._tensors_set = True
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
        max_len = self.max_sequence_length

        obs_shape = self.observations[decoy_interval][0].shape
        act_shape = self.actions[decoy_interval][0].shape if self.actions[decoy_interval].numel() > 0 else (1,)

        obs_batch = torch.zeros((batch_size, max_len, *obs_shape), dtype=torch.float32, device=self._device)
        next_obs_batch = torch.zeros_like(obs_batch)
        action_batch = torch.zeros((batch_size, max_len, *act_shape), dtype=torch.float32, device=self._device)
        reward_batch = torch.zeros((batch_size, max_len, 1), dtype=torch.float32, device=self._device)
        done_batch = torch.zeros((batch_size, max_len, 1), dtype=torch.bool, device=self._device)
        visible_batch = torch.zeros((batch_size, max_len, 1), dtype=torch.bool, device=self._device)

        # This loop is vectorized for performance using advanced indexing
        # 1. Get sequence start indices and lengths from the sampled indices
        seq_info_batch = self.sequence_info[decoy_interval][idxs]
        starts = seq_info_batch[:, 0]
        actual_lens = seq_info_batch[:, 1]

        # 2. Create index grids for fetching all data in one go
        # `arange` creates the sequence offsets (0, 1, ..., max_len-1)
        # `starts.unsqueeze(1)` broadcasts the start indices for each sequence
        # Shape of `indices`: (batch_size, max_len)
        base_indices = torch.arange(max_len, device=self._device).unsqueeze(0)
        indices = starts.unsqueeze(1) + base_indices
        next_indices = indices + 1

        # 3. Create a mask to select only the valid, non-padded indices
        # Shape of `fetch_mask`: (batch_size, max_len)
        fetch_mask = base_indices < actual_lens.unsqueeze(1)

        # 4. Fetch all data using the masks and indices
        obs_batch[fetch_mask] = self.observations[decoy_interval][indices[fetch_mask]]
        action_batch[fetch_mask] = self.actions[decoy_interval][indices[fetch_mask]]
        reward_batch[fetch_mask] = self.rewards[decoy_interval][indices[fetch_mask]].unsqueeze(-1)
        done_batch[fetch_mask] = self.dones[decoy_interval][indices[fetch_mask]].unsqueeze(-1)
        visible_batch[fetch_mask] = self.visible_states[decoy_interval][indices[fetch_mask]].unsqueeze(-1)

        # 5. --- CORRECTED NEXT_OBS LOGIC ---
        # The next_obs sequence is the obs sequence shifted by one.
        # We use `next_indices` and the same `fetch_mask`. This works because the buffer
        # should have a terminal zero-pad, making the last `next_index` valid.
        next_obs_batch[fetch_mask] = self.observations[decoy_interval][next_indices[fetch_mask]]

        # 6. The training mask is just the fetch_mask with an added dimension
        mask_batch = fetch_mask.unsqueeze(-1)

        # This can be derived from the visible_batch after fetching
        next_visible_batch = torch.roll(visible_batch, shifts=-1, dims=1)
        next_visible_batch[:, -1] = False  # Last step has no defined next step

        return (obs_batch, action_batch.unsqueeze(-1) if action_batch.dim() == 2 else action_batch,
                reward_batch, next_obs_batch, done_batch,
                visible_batch, next_visible_batch, mask_batch)


class SaveEachBestCallback(BaseCallback):
    """Called by EvalCallback when a new best model is found."""
    def __init__(self, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0

    def _on_step(self) -> bool:
        # This is triggered by EvalCallback when there's a new best
        self.idx += 1
        best_mean = getattr(self.parent, "best_mean_reward", None)  # provided by EvalCallback
        if best_mean is None:
            fname = f"best_{self.idx:03d}_steps={self.num_timesteps}.zip"
        else:
            fname = f"best_{self.idx:03d}_steps={self.num_timesteps}_mean={best_mean:.2f}.zip"
        path = self.save_dir / fname
        self.model.save(str(path))
        if self.verbose:
            print(f"[SaveEachBest] Saved: {path}")
        return True


class EnvironmentEvaluator:
    def __init__(self, env, n_trials: int, min_scale_rewards: float = 0.0, max_scale_rewards: float = 1.0,
                 running_average_obs: bool = False):
        assert n_trials > 0, "n_trials must be positive"
        assert max_scale_rewards > min_scale_rewards, "max_scale_rewards must be greater than min_scale_rewards"
        # Scale rewards to [0, 1]
        self.env = env
        self.n_trials = n_trials
        self.running_average_obs = running_average_obs

        self.min_scale = min_scale_rewards
        self.max_scale = max_scale_rewards

    def __call__(self, algo) -> float:
        mean_returns = []
        running_average_obs = deque(maxlen=12)
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
                                                        hidden_state=hidden_state)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

            # Scale total reward to [0, 1]
            total_reward = (total_reward - self.min_scale) / (self.max_scale - self.min_scale)
            mean_returns.append(total_reward)

        return float(np.mean(mean_returns)), float(np.std(mean_returns) / np.sqrt(self.n_trials))


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
