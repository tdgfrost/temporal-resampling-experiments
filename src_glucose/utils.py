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
import json
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

        self.max_rewards_scale = None
        self.min_rewards_scale = None
        self.buffer_size = buffer_size
        self.env = env
        self._tensors_set = False
        self._device = None

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
        ]:
            save_dict = {i: list(value[i]) for i in range(3)}
            np.savez(os.path.join(path, f'{key}.npz'),
                     **{str(k): v for k, v in save_dict.items()})

        # Also save min/max rewards scale
        np.savez(os.path.join(path, 'rewards_scale.npz'),
                 **{'min': self.min_rewards_scale, 'max': self.max_rewards_scale})

        # Mark saving as complete
        with open(os.path.join(path, 'COMPLETE'), 'w') as f:
            f.close()

    def load(self, path: str):
        for key, value in [
            ('observations', self.observations),
            ('actions', self.actions),
            ('rewards', self.rewards),
            ('dones', self.dones),
        ]:
            loaded = np.load(os.path.join(path, f'{key}.npz'), allow_pickle=True)
            for k in loaded.files:
                value[int(k)] = deque(loaded[k], maxlen=self.buffer_size)

        # Load min/max rewards scale
        loaded_scale = np.load(os.path.join(path, 'rewards_scale.npz'), allow_pickle=True)
        self.min_rewards_scale = float(loaded_scale['min'])
        self.max_rewards_scale = float(loaded_scale['max'])

    def reset(self, seed: int = None):
        obs, info = self.env.reset(seed=seed)
        ep_buffer = self._reset_ep_buffer(obs, info)
        return obs, info, ep_buffer

    @staticmethod
    def _reset_ep_buffer(obs, info):
        return {
            'obs': [obs], 'decoy_obs': [info['obs'][0]],
            'action': [], 'decoy_action': [],
            'reward': [], 'decoy_reward': [],
            'done': [], 'decoy_done': [],
        }

    def fill_buffer(self, model, n_frames: int = 1_000, seed: int = None, with_random: bool = True, rand_p: float = 0.05):
        with tqdm(total=n_frames, desc="Progress", mininterval=2.0) as pbar:
            frame_count = 0
            if seed is None:
                seed = 123

            obs, info, ep_buffer = self.reset(seed=seed)
            model.set_random_seed(seed)

            while frame_count < n_frames:
                done = False
                while not done:
                    action, _ = model.predict(obs)
                    # if with_random and np.random.random() < rand_p:
                        # action = np.random.randint(0, 4)
                    obs, reward, term, trunc, info = self.env.step(action)
                    done = term or trunc

                    self.update_episode_buffer(obs, action, reward, done, info, ep_buffer)

                    if done:
                        ep_buffer['obs'] = ep_buffer['obs'][:-1]
                        ep_buffer['decoy_obs'] = ep_buffer['decoy_obs'][:-1]
                        obs, info = self.env.reset(seed=seed + frame_count)

                    pbar.update(1)
                    frame_count += 1
                    model.set_random_seed(seed + frame_count)

                # Add ep_buffer to permanent buffer
                self.update_permanent_buffer(ep_buffer)
                # Reset ep_buffer and add 'obs' to it
                ep_buffer = self._reset_ep_buffer(obs, info)

            # Add a garbage all-zeros "final obs"
            for i in [0, 1, 2]:
                self.observations[i] += [np.zeros_like(self.observations[0][0])]

            # Normalise rewards from 0 to 1
            for i in [0, 1, 2]:
                rewards = np.array(self.rewards[i])
                min_r, max_r = rewards.min(), rewards.max()
                if max_r > min_r:
                    norm_rewards = (rewards - min_r) / (max_r - min_r)
                else:
                    norm_rewards = rewards - min_r  # All zeros
                self.rewards[i] = deque(norm_rewards.tolist(), maxlen=self.buffer_size)
                self.min_rewards_scale = min_r
                self.max_rewards_scale = max_r

    def set_to_tensors(self, device: str = 'cpu'):
        if self._tensors_set:
            if self._device == device:
                return
            for i in [0, 1, 2]:
                for arr in [self.observations, self.actions, self.rewards, self.dones]:
                    arr[i] = arr[i].to(device)
        else:
            for i in [0, 1, 2]:
                for arr in [self.observations, self.actions, self.rewards, self.dones]:
                    arr[i] = torch.from_numpy(np.array(arr[i])).to(device)

        self._tensors_set = True
        self._device = device

    @staticmethod
    def update_episode_buffer(obs, action: Union[int, np.ndarray], reward: float, done: bool, info: dict, ep_buffer: dict):
        for key in ['obs', 'action', 'reward', 'done']:
            ep_buffer[f'decoy_{key}'] += info[key]

        ep_buffer['obs'] += [obs]
        ep_buffer['action'] += [action]
        ep_buffer['reward'] += [reward]
        ep_buffer['done'] += [done]

    def update_permanent_buffer(self, ep_buffer: dict):
        for i, decoy_maybe in [(0, ''), (1, 'decoy_')]:
            for arr, key in [
                (self.observations, 'obs'), (self.actions, 'action'), (self.rewards, 'reward'), (self.dones, 'done')]:
                arr[i] += ep_buffer[f'{decoy_maybe}{key}']

        # Update the decoy actions (every second step)
        # - For basal insulin, this involves taking the average of the two actions
        actions = [
            np.mean(ep_buffer['decoy_action'][idx: idx + 2])
            for idx in range(0, len(ep_buffer['decoy_action']), 2)
        ]

        obs, rewards, dones = [
            [ep_buffer[f'decoy_{key}'][idx] for idx in range(0, len(ep_buffer[f'decoy_{key}']), 2)]
            for key in ['obs', 'reward', 'done']
        ]

        if not dones[-1]:
            rewards[-1] = ep_buffer['reward'][-1]
            dones[-1] = ep_buffer['done'][-1]
        assert dones[-1], "Last done flag should be True"
        self.observations[2] += obs
        self.actions[2] += actions
        self.rewards[2] += rewards
        self.dones[2] += dones

    def sample_transition_batch(self, batch_size: int = 32, decoy_interval: int = 0):
        assert self._tensors_set, "Replay buffer must be set to tensors first using .set_to_tensors(model)"
        idxs = torch.randint(0, len(self.observations[decoy_interval]) - 1, (batch_size,), device=self._device)

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
    def __init__(self, env, n_trials: int, min_scale_rewards: float = 0.0, max_scale_rewards: float = 1.0):
        assert n_trials > 0, "n_trials must be positive"
        assert max_scale_rewards > min_scale_rewards, "max_scale_rewards must be greater than min_scale_rewards"
        # Scale rewards to [0, 1]
        self.env = env
        self.n_trials = n_trials
        self.min_scale = min_scale_rewards
        self.max_scale = max_scale_rewards

    def __call__(self, algo) -> float:
        mean_returns = []
        for _ in range(self.n_trials):
            obs, info = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                with torch.no_grad():
                    action = algo.predict(np.expand_dims(obs, 0))
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
