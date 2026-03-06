from typing import Union, Callable, Tuple
import numpy as np
import torch
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from collections import deque
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import argparse
import inquirer
import os


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

        self.buffer_size = buffer_size
        self.env = env
        self._tensors_set = False
        self._device = None

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

            obs, info, ep_buffer = self.reset(seed=int(seed))
            model.set_random_seed(seed)

            while frame_count < n_frames:
                done = False
                while not done:
                    action, _ = model.predict(obs)
                    if with_random and np.random.random() < rand_p:
                        action = np.random.randint(0, 4)
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

        obs, actions, rewards, dones = [
            [ep_buffer[f'decoy_{key}'][idx] for idx in range(0, len(ep_buffer[f'decoy_{key}']), 2)]
            for key in ['obs', 'action', 'reward', 'done']
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
        if obs.shape[-1] in (1, 5):
            return obs[:, -1, -1, 0].unsqueeze(-1).long()
        return obs[:, 0, -1, -1].unsqueeze(-1).long()


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
    def __init__(self, env, n_trials: int):
        self.env = env
        self.n_trials = n_trials

    def __call__(self, algo, seed=None) -> float:
        mean_returns = []
        if seed is not None:
            _, _ = self.env.reset(seed=seed)
        else:
            _, _ = self.env.reset()

        for _ in range(self.n_trials):
            obs, info = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                with torch.no_grad():
                    action = algo.predict(np.expand_dims(obs, 0))
                    if isinstance(action, tuple):
                        action = action[0]
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

            mean_returns.append(total_reward)

        return float(np.mean(mean_returns)), float(np.std(mean_returns) / np.sqrt(self.n_trials))


class ParallelEnvironmentEvaluator:
    def __init__(self,
                 env_creator: Callable[[], gym.Env],
                 n_eval_episodes: int,
                 n_eval_envs: int = 4):
        """
        :param env_creator: A function that returns a gymnasium Environment.
                            (e.g. lambda: gym.make('MyEnv-v0'))
        :param n_eval_episodes: Total number of episodes to collect for the statistic.
        :param n_eval_envs: Number of parallel environments to run.
        """
        self.env_creator = env_creator
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.vec_env = AsyncVectorEnv([self.env_creator for _ in range(self.n_eval_envs)])

    def __call__(self, algo, seed: int = None) -> Tuple[float, float]:
        """
        Runs the evaluation.
        Returns: (mean_reward, standard_error)
        """
        # 2. Reset Envs
        if seed is not None:
            # Create a list of seeds, one for each env
            rng = np.random.default_rng(seed)
            env_seeds = rng.integers(0, 2 ** 32 - 1, size=self.n_eval_envs).tolist()
            obs, info = self.vec_env.reset(seed=env_seeds)
        else:
            obs, info = self.vec_env.reset()

        # 3. Variables to track progress
        completed_episode_rewards = []
        current_episode_rewards = np.zeros(self.n_eval_envs)

        # We use a progress bar for better visibility
        pbar = tqdm(total=self.n_eval_episodes, desc="Evaluating", unit="ep", leave=False)

        # 4. Evaluation Loop
        while len(completed_episode_rewards) < self.n_eval_episodes:

            # --- Predict ---
            with torch.no_grad():
                # obs is already (n_envs, obs_dim), so no expand_dims needed
                # assuming algo.predict handles batch inputs
                action = algo.predict(obs)

                # Handle tuple actions (common in some custom algos)
                if isinstance(action, tuple):
                    action = action[0]

            # Ensure action is a numpy array for the env
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # --- Step ---
            # vec_env.step automatically resets sub-envs that are done
            obs, rewards, terminated, truncated, infos = self.vec_env.step(action)
            dones = terminated | truncated

            current_episode_rewards += rewards

            # --- Handle Completions ---
            if np.any(dones):
                for i, done in enumerate(dones):
                    if done:
                        # Store the result
                        completed_episode_rewards.append(current_episode_rewards[i])
                        pbar.update(1)

                        # Reset accumulator for this specific environment
                        current_episode_rewards[i] = 0.0

                        # Optimization: Stop early if we have enough data
                        if len(completed_episode_rewards) >= self.n_eval_episodes:
                            break

        pbar.close()

        # 5. Calculate Statistics
        # Slice to exactly n_eval_episodes in case we over-collected slightly
        results = completed_episode_rewards[:self.n_eval_episodes]

        mean_return = float(np.mean(results))
        std_error = float(np.std(results) / np.sqrt(self.n_eval_episodes))

        return mean_return, std_error


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
