from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.utils import RecordConstructorArgs
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from minigrid.wrappers import ImgObsWrapper, Wrapper
from gymnasium.envs.registration import register
from datetime import datetime, timedelta
from simglucose.simulation.scenario import CustomScenario


SAMPLE_TIME = 60.0  # minutes
for i in range(1, 6):
    register(
        id=f"simglucose/adult{i}-v0",
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        kwargs={"patient_name": f"adult#00{i}"},
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

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        i = 1  # np.random.randint(1, 6)
        identity = f"simglucose/adult{i}-v0"
        env = gym.make(identity, max_episode_steps=(24 * 60) // SAMPLE_TIME, **self.kwargs)
        super().__init__(env)

    def reset(self, **kwargs):
        # Rebuild env each reset
        self.env.close()  # cleanup
        i = 1  # np.random.randint(1, 6)
        identity = f"simglucose/adult{i}-v0"
        self.env = gym.make(identity, max_episode_steps=(24 * 60) // SAMPLE_TIME, **self.kwargs)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class FixedScaler(ObservationWrapper):

    def __init__(self, env, **kwargs):
        self.kwargs = kwargs
        ObservationWrapper.__init__(self, env)

    def observation(self, obs):
        # Max BG = 600, min BG = 10
        # Max insulin = 30, min insulin = 0
        # Max CHO = 300, min = 0
        obs[0] = (obs[0] - 10) / (600 - 10)  # BG
        obs[1] = obs[1] / 30.0                # insulin
        obs[2] = obs[2] / 300.0               # CHO
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
        self.last_waiting_period = 2
        self.next_waiting_period = 0
        assert 0 <= forced_interval <= 1, "Forced interval must be 0 or 1"
        self.forced_interval = forced_interval

    def reset(self, *args, **kwargs) -> np.ndarray:
        self.last_action = None
        self.steps_until_action_available = 0
        self.next_waiting_period = 0
        obs, info = self.env.reset(*args, **kwargs)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.steps_until_action_available == 0:
            # Do the action
            obs, reward, term, trunc, info = self.env.step(action)
            # flip the steps, calculate steps remaining
            self._flip_step_modes(action)
        else:
            # Repeat the last action
            obs, reward, term, trunc, info = self.env.step(self.last_action)

            # Deduce 1 step from steps remaining
            self.steps_until_action_available -= 1

        return obs, reward, term, trunc, info

    def _flip_step_modes(self, action: Any):
        self.steps_until_action_available = self.next_waiting_period
        # self.next_waiting_period = np.random.choice([0, 5])
        self.next_waiting_period, self.last_waiting_period = self.last_waiting_period, self.next_waiting_period
        self.last_action = action


class RepeatFlagChannel(RecordConstructorArgs, ObservationWrapper):
    """
    Original obs shape (47,). Append a 1-channel flag at the start to make (48).
    0 -> next action repeats once; 1 -> next action repeats twice.
    """
    def __init__(self, env, use_flag: bool = True):
        RecordConstructorArgs.__init__(self)
        ObservationWrapper.__init__(self, env)

        assert isinstance(env.observation_space, spaces.Box)
        low, high = env.observation_space.low, env.observation_space.high
        self.state_dim = (low.shape[0] + 2,)
        self.dtype = low.dtype
        low, high = np.concatenate([[0, 0], low]), np.concatenate([[1, 2], high])
        self.observation_space = spaces.Box(
            low=low, high=high, shape=self.state_dim, dtype=self.dtype
        )
        self.use_flag = use_flag

    def observation(self, obs):
        # Concat flag (0/1) to the start of the channels
        # - always set to 0 if use_flag = False
        steps_left = self.env.get_wrapper_attr("steps_until_action_available") if self.use_flag else 0
        waiting_period = self.env.get_wrapper_attr("next_waiting_period") if self.use_flag else 0
        return np.concatenate([[steps_left, waiting_period], obs], axis=-1)


def make_glucose_env(*, no_interim_rewards: bool = False, gamma: float = 1.0, forced_interval: int = 0,
                     use_flag: bool = True, use_scaling: bool = False, **kwargs):
    env = T1DPatientEnv(**kwargs)
    env = SampleTimeWrapper(env)
    if use_scaling:
        env = NormalizeReward(env, gamma=gamma)
    if no_interim_rewards:
        env = EpisodeRewardsOnly(env)
    env = AlternateStepWrapper(env, forced_interval=forced_interval)
    env = FixedScaler(env)
    env = RepeatFlagChannel(env, use_flag=use_flag)     # +1 channel flag
    return env
