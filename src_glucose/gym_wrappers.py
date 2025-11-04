from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.utils import RecordConstructorArgs
from gymnasium.wrappers import NormalizeReward
from minigrid.wrappers import Wrapper
from gymnasium.envs.registration import register
import random

SAMPLE_TIME = 10.0  # minutes
AGGREGATE_WINDOW_SIZE = 24  # 24 * 10 minutes = 240 minutes
TOTAL_SIZE = 12  # Set irregular sampling from 10 minutes to 120 minutes (12 * 10 minutes)

# Scaling parameters
INSULIN_SCALE = 30.0
CHO_SCALE = 300.0

# Get our adult patients
for i in range(1, 11):
    register(
        id=f"simglucose/adult{i}-v0",
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        kwargs={"patient_name": f"adult#0{i:02}"},
    )


class SampleTimeWrapper(RecordConstructorArgs, Wrapper):
    """
    Allow adjustment of sample time
    """

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Update sample time
        self.env.unwrapped.env.env.sample_time = SAMPLE_TIME
        info['sample_time'] = SAMPLE_TIME
        return obs, info


class EpisodeRewardsOnly(RecordConstructorArgs, Wrapper):
    """
    Set all intermediate rewards to 0.
    At the end of the episode, give the sum of all rewards received.
    """

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)
        self.episode_rewards = []

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.episode_rewards.append(rew)
        if terminated or truncated:
            total_reward = np.sum(self.episode_rewards) if self.episode_rewards else 0.0
            reward = total_reward
            self.episode_rewards = []
        else:
            reward = 0.0
        return obs, reward, terminated, truncated, info


class T1DPatientEnv(Wrapper):

    def __init__(self, use_test_ids: bool = False, **kwargs):
        self.kwargs = kwargs
        self._id_choices = [1, 2, 3, 4, 5, 6, 7, 8] if not use_test_ids else [9, 10]
        id_choice = np.random.choice(self._id_choices)
        identity = f"simglucose/adult{id_choice}-v0"
        env = gym.make(identity, max_episode_steps=(48 * 60) // SAMPLE_TIME, **self.kwargs)
        super().__init__(env)

    def reset(self, **kwargs):
        # Rebuild env each reset
        self.env.close()  # cleanup
        id_choice = np.random.choice(self._id_choices)
        identity = f"simglucose/adult{id_choice}-v0"
        self.env = gym.make(identity, max_episode_steps=(48 * 60) // SAMPLE_TIME, **self.kwargs)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def get_time(self):
        time = self.unwrapped.env.env.time
        hour_float = (time.day - 1) * 24 + time.hour + time.minute / 60.0
        return hour_float


class FixedScaler(ObservationWrapper):

    def __init__(self, env, **kwargs):
        self.kwargs = kwargs
        ObservationWrapper.__init__(self, env)

        low, high = env.observation_space.low, env.observation_space.high
        self.state_dim = (low.shape[0],)
        self.dtype = low.dtype
        self.observation_space = spaces.Box(
            low=low, high=10 * np.ones_like(high), shape=self.state_dim, dtype=self.dtype
        )

    def observation(self, obs):
        # Max BG = 600, min BG = 10
        # Max insulin = 30, min insulin = 0
        # Max CHO = 300, min = 0
        obs[0] = (obs[0] - 10) / (600 - 10)  # BG
        obs[1] = obs[1] / INSULIN_SCALE  # insulin
        obs[2] = obs[2] / CHO_SCALE  # CHO
        return obs


class AlternateStepWrapper(RecordConstructorArgs, Wrapper):
    """
    A wrapper that may perform additional
    'bonus' steps using the same action.
    """

    def __init__(self, env: gym.Env, forced_interval: int = 0) -> None:
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)
        self.last_action = None
        self.steps_until_action_available = 0
        self.next_waiting_period = 0
        self.hour_float = None
        assert 0 <= forced_interval <= 1, "Forced interval must be 0 or 1"
        self.forced_interval = forced_interval

    def reset(self, *args, **kwargs) -> np.ndarray:
        self.last_action = None
        self.steps_until_action_available = 0
        self.next_waiting_period = 0
        obs, info = self.env.reset(*args, **kwargs)
        self._apply_rules(obs)
        info['steps_until_action_available'] = self.steps_until_action_available
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.steps_until_action_available == 0 or self.forced_interval == 1:
            # Do the action
            obs, reward, term, trunc, info = self.env.step(action)
            # flip the steps, calculate steps remaining
            self._flip_step_modes(action, obs)
            info['steps_until_action_available'] = self.steps_until_action_available
        else:
            # Repeat the last action
            obs, reward, term, trunc, info = self.env.step(self.last_action)

            # Deduce 1 step from steps remaining
            self.steps_until_action_available -= 1
            info['steps_until_action_available'] = self.steps_until_action_available

        return obs, reward, term, trunc, info

    def _flip_step_modes(self, action: Any, obs):
        self._apply_rules(obs)
        self.last_action = action

    def _apply_rules(self, obs):
        """
        Let's establish rules:
        - if blood glucose goes <70, wake up and do something about it - check every 10 minutes
        - if blood glucose goes >300, wake up and do something about it - check every 10 minutes
        Otherwise:
        - 2200 - 0600 - repeat last action throughout
        - 0600 - 2200 - Lognormal sampling
        """
        current_bg = self._get_current_bg(obs)
        if False: #current_bg < 54 or current_bg > 450:
            self.steps_until_action_available = 0
            self.next_waiting_period = 0
            return

        self.steps_until_action_available = self.next_waiting_period
        # self.next_waiting_period = int(np.clip(np.rint(np.random.normal(12.0, 2)), 6, TOTAL_SIZE)) - 1
        # self.next_waiting_period = int(np.random.choice([1, 3, 6, 9, 12], p=[0.1, 0.1, 0.1, 0.1, 0.6])) - 1
        self.next_waiting_period = random.randint(1, TOTAL_SIZE) - 1

    @staticmethod
    def _get_current_bg(obs):
        return obs[0] * 590 + 10


class RepeatFlagChannel(RecordConstructorArgs, ObservationWrapper):
    """
    Original obs shape (47,). Append a 1-channel flag at the start to make (48).
    0 -> next action repeats once; 1 -> next action repeats twice.
    """

    def __init__(self, env):
        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)

        assert isinstance(env.observation_space, spaces.Box)
        low, high = env.observation_space.low, env.observation_space.high
        self.state_dim = (low.shape[0] + 1,)
        self.dtype = low.dtype
        low, high = np.concatenate([low, [0]]), np.concatenate([high, [1]])
        self.observation_space = spaces.Box(
            low=low, high=high, shape=self.state_dim, dtype=self.dtype
        )

    def observation(self, obs):
        hour_float = self.env.get_wrapper_attr('get_time')() / 48
        return np.concatenate([obs, [hour_float]], axis=-1)


class EnforcePPOWrapper(Wrapper):
    """
    A wrapper that may perform additional
    'bonus' steps using the same action.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99) -> None:
        Wrapper.__init__(self, env)
        self._gamma = gamma
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low[np.newaxis],
            high=env.observation_space.high[np.newaxis],
            shape=(1,) + env.observation_space.shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, *args, **kwargs) -> np.ndarray:
        obs, info = self.env.reset(*args, **kwargs)
        info['steps_taken'] = 1
        return np.expand_dims(obs, 0), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, trunc, info = self.env.step(action)
        obs_lst = [obs]
        done = term or trunc
        steps_taken = 1
        while self.env.get_wrapper_attr("steps_until_action_available") != 0 and not done:
            obs, reward_step, term, trunc, info = self.env.step(action)
            obs_lst.append(obs)
            reward += reward_step * self._gamma ** steps_taken
            done = term or trunc
            steps_taken += 1

        info['steps_taken'] = steps_taken

        obs = np.stack(obs_lst, axis=0)

        return obs, reward, term, trunc, info


def make_glucose_env(*, use_test_ids: bool = False, no_interim_rewards: bool = False, gamma: float = 1.0,
                     forced_interval: int = 0, use_scaling: bool = False, **kwargs):
    env = T1DPatientEnv(use_test_ids, **kwargs)
    env = SampleTimeWrapper(env)
    if use_scaling:
        env = NormalizeReward(env, gamma=gamma)
    if no_interim_rewards:
        env = EpisodeRewardsOnly(env)
    env = FixedScaler(env)
    env = AlternateStepWrapper(env, forced_interval=forced_interval)
    env = RepeatFlagChannel(env)
    return env
