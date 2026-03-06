from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.utils import RecordConstructorArgs
from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import ImgObsWrapper, Wrapper, FullyObsWrapper
from datetime import datetime
import random


class AlternateStepWrapper(RecordConstructorArgs, Wrapper):
    """
    A wrapper that may perform additional
    'bonus' steps using the same action.
    """

    def __init__(self, env: gym.Env, max_steps: int = 100, forced_interval: int = 0, fixed_reward: bool = True) -> None:
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)
        # super().__init__(env)
        self.last_step_mode = 0
        self.current_step_mode = 0
        self.step_count = 0
        self.max_steps = max_steps
        self.fixed_reward = fixed_reward
        assert 0 <= forced_interval <= 1, "Forced interval must be 0 or 1"
        self.forced_interval = forced_interval

    def reset(self, *args, **kwargs) -> np.ndarray:
        self.last_step_mode = 0
        self.current_step_mode = 0
        self.step_count = 0
        obs, info = self.env.reset(*args, **kwargs)
        info['obs'] = [obs]
        info['action'] = []
        info['reward'] = []
        info['done'] = []
        info = self._take_no_additional_steps(info)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Do our first genuine environment step
        obs, reward, term, trunc, info = self._take_first_step(action)

        # Decide if we are forcing 1-step intervals
        if self.forced_interval:
            info = self._take_no_additional_steps(info)
            info['done'][-1] = term or trunc
            return obs, reward, (term or trunc), False, info

        # Take 1-step only (or if environment has already terminated)
        if term or self.current_step_mode == 0:
            # Update step_mode for next observation
            self._flip_step_modes()

            info = self._take_no_additional_steps(info)
            info['done'][-1] = term or trunc
            return obs, reward, (term or trunc), False, info

        # Take 3-steps
        elif self.current_step_mode == 1:
            # Update step_mode for next observation
            self._flip_step_modes()

            # Take another step (second)
            obs, reward, term, _, info = self._take_another_step(action, info)
            if term:
                info['done'][-1] = term or trunc
                return obs, reward, (term or trunc), False, info

            # Take another step (third)
            obs, reward, term, _, info = self._take_another_step(action, info)
            info['done'][-1] = term or trunc
            return obs, reward, (term or trunc), False, info

        else:
            raise ValueError(f"Invalid step_mode: {self.step_mode}")

    def _flip_step_modes(self):
        self.last_step_mode = self.current_step_mode
        # self.current_step_mode = 1 - self.current_step_mode
        self.current_step_mode = random.choice([0, 1])

    def _take_first_step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, _, info = self.env.step(action)
        self.step_count += 1
        trunc = self._get_trunc()
        reward = self._update_step_reward(reward)
        info['obs'] = [obs]
        info['action'] = [action]
        info['reward'] = [reward]
        info['done'] = [False]
        return obs, reward, term, trunc, info

    @staticmethod
    def _take_no_additional_steps(info):
        info['bonus_step_taken'] = False
        return info

    def _take_another_step(self, action: Any, base_info: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Take additional step
        base_info['bonus_step_taken'] = True

        obs, reward, term, _, new_info = self.env.step(action)

        # Update return variables
        base_info, reward = self._update_info_and_reward(base_info, new_info, reward)
        base_info['obs'].append(obs)
        base_info['action'].append(action)
        base_info['reward'].append(reward)
        base_info['done'].append(False)
        return obs, reward, term, _, base_info

    def _update_info_and_reward(self, base_info, new_info, reward):
        base_info.update(new_info)
        reward = self._update_step_reward(reward)
        return base_info, reward

    def _get_trunc(self):
        # This is used to set term, NOT trunc
        if self.step_count >= self.max_steps:
            return True
        return False

    def _update_step_reward(self, reward: float) -> float:
        # Simplify the reward to per-step basis
        if reward > 0:
            return 1.0 if self.fixed_reward else float(reward)
        return 0.0


class RepeatFlagChannel(RecordConstructorArgs, ObservationWrapper):
    """
    Original obs shape (5, 5, 3). Append a 1-channel flag to make (5, 5, 4).
    0 -> next action repeats once; 1 -> next action repeats twice.
    """
    def __init__(self, env, use_flag: bool = True):
        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)

        assert isinstance(env.observation_space, spaces.Box)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c + 1), dtype=np.uint8
        )
        self.use_flag = use_flag

    def observation(self, obs):
        # Concat flag (0/1) to the start of the channels
        # - always set to 0 if use_flag = False
        val = 1 if self.env.get_wrapper_attr("last_step_mode") == 1 and self.use_flag else 0 # 0 for no repeat, 1 for repeat
        flag = np.full((obs.shape[0], obs.shape[1], 1), val, dtype=np.uint8)
        return np.concatenate([flag, obs], axis=-1)


class RecordableImgObsWrapper(RecordConstructorArgs, ImgObsWrapper):
    """
    A version of ImgObsWrapper that records constructor args.
    """
    def __init__(self, env):
        RecordConstructorArgs.__init__(self)
        ImgObsWrapper.__init__(self, env)
        assert isinstance(env.observation_space['image'], spaces.Box)
        h, w, c = env.observation_space["image"].shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c + 1), dtype=np.uint8
        )

    def observation(self, obs):
        target_shape = obs['image'].shape
        target_dtype = obs['image'].dtype
        return np.concatenate((np.full(target_shape[:-1] + (1,), fill_value=obs['direction'], dtype=target_dtype),
                               obs["image"]), -1)


class DecoyObsWrapper(RecordConstructorArgs, Wrapper):
    def __init__(self, env):
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info = self._fill_obs(info)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = self.env.step(action)
        info = self._fill_obs(info)
        return obs, rew, term, trunc, info

    @staticmethod
    def _fill_obs(info):
        for idx, vanilla_obs in enumerate(info['obs']):
            obs, direction = vanilla_obs['image'], vanilla_obs['direction']
            target_shape = obs.shape[:-1] + (1,)
            # Concat flag to the start of the channels
            obs = np.concatenate((np.full(target_shape, fill_value=direction, dtype=obs.dtype),
                                  obs), -1)

            # Always set decoy obs flag to zero and concat to the start of the channels
            flag = np.full(target_shape, 0, dtype=obs.dtype)
            obs = np.concatenate([flag, obs], axis=-1)
            info['obs'][idx] = obs
        return info


folder_name = datetime.now().strftime("%Y%m%d-%H%M%S")
current_video_folder = f'../logs/videos/{folder_name}/'


def all_episodes_trigger(x):
    return True


def make_video_lavastep_env(*, max_steps=100, forced_interval: int = 0, use_flag: bool = False,
                            fixed_reward: bool = True, video_folder=current_video_folder,
                            episode_trigger=all_episodes_trigger, step_trigger=None, video_length=0,
                            name_prefix='rl-video', disable_logger=False, **kwargs):
    env_name = "MiniGrid-LavaGapS6-v0"
    env = gym.make(env_name, max_episode_steps=None, **kwargs)
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=episode_trigger,
                      step_trigger=step_trigger, video_length=video_length, name_prefix=name_prefix,
                      disable_logger=disable_logger)
    # env = FullyObsWrapper(env)
    env = AlternateStepWrapper(env, max_steps=max_steps, forced_interval=forced_interval, fixed_reward=fixed_reward)
    env = RecordableImgObsWrapper(env)         # (H,W,C) uint8 image
    env = RepeatFlagChannel(env, use_flag=use_flag)     # +1 channel flag
    env = DecoyObsWrapper(env)
    return env


def make_lavastep_env(*, max_steps=100, forced_interval: int = 0, use_flag: bool = True,
                      fixed_reward: bool = True, **kwargs):
    env_name = "MiniGrid-LavaGapS6-v0"
    env = gym.make(env_name, max_episode_steps=None, **kwargs)
    # env = FullyObsWrapper(env)
    env = AlternateStepWrapper(env, max_steps=max_steps, forced_interval=forced_interval, fixed_reward=fixed_reward)
    env = RecordableImgObsWrapper(env)         # (H,W,C) uint8 image
    env = RepeatFlagChannel(env, use_flag=use_flag)     # +1 channel flag
    env = DecoyObsWrapper(env)
    return env


