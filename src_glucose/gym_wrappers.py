from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.utils import RecordConstructorArgs
from gymnasium.wrappers import NormalizeObservation
from minigrid.wrappers import ImgObsWrapper, Wrapper
from gymnasium.envs.registration import register
from datetime import datetime, timedelta
from simglucose.simulation.scenario import CustomScenario


SAMPLE_TIME = 20.0  # minutes
for i in range(1, 6):
    register(
        id=f"simglucose/adult{i}-v0",
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        kwargs={"patient_name": f"adult#00{i}"},
    )


class CombinedObservationWrapper(RecordConstructorArgs, Wrapper):
    """
    Replace observations with the continuous state vector carried in `info[state_key]`.
    You must supply the expected length so we can set observation_space up-front.
    """
    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)
        self.state_key = 'patient_state'

        # Get our info obs
        patient_state_example = self.env.reset()[1]['patient_state']
        self.state_dim = patient_state_example.shape
        self.dtype = patient_state_example.dtype

        # Advertise the new observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.state_dim, dtype=self.dtype
        )

    def observation(self, observation):  # not used; we override via step/reset directly
        return observation

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Update sample time
        self.env.unwrapped.env.env.sample_time = SAMPLE_TIME
        info['sample_time'] = SAMPLE_TIME
        state = self._extract_state(info)
        return state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._extract_state(info)
        return state, reward, terminated, truncated, info

    def _extract_state(self, info):
        if self.state_key not in info:
            raise KeyError(f"Expected '{self.state_key}' in info, got keys: {list(info.keys())}")
        state = np.asarray(info[self.state_key], dtype=self.dtype)
        if state.shape != self.state_dim:
            raise ValueError(f"{self.state_key} shape {state.shape} != {self.state_dim}")
        return state


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
        i = np.random.randint(1, 6)
        identity = f"simglucose/adult{i}-v0"
        env = gym.make(identity, max_episode_steps=(24 * 60) // SAMPLE_TIME, **self.kwargs)
        super().__init__(env)

    def reset(self, **kwargs):
        # Rebuild env each reset
        self.env.close()  # cleanup
        i = np.random.randint(1, 6)
        identity = f"simglucose/adult{i}-v0"
        self.env = gym.make(identity, max_episode_steps=(24 * 60) // SAMPLE_TIME, **self.kwargs)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class AlternateStepWrapper(RecordConstructorArgs, Wrapper):
    """
    A wrapper that may perform additional
    'bonus' steps using the same action.
    """

    def __init__(self, env: gym.Env, forced_interval: int = 0, gamma: float = 1.0) -> None:
        RecordConstructorArgs.__init__(self)
        Wrapper.__init__(self, env)
        # super().__init__(env)
        self.last_step_mode = 0
        self.current_step_mode = 0
        assert 0 <= forced_interval <= 1, "Forced interval must be 0 or 1"
        self.forced_interval = forced_interval
        self.gamma = gamma
        assert 0.0 < gamma <= 1.0, "Gamma must be in (0.0, 1.0]"

    def reset(self, *args, **kwargs) -> np.ndarray:
        self.last_step_mode = 0
        self.current_step_mode = 0
        # self.step_count = 0
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
        done = term or trunc

        # Decide if we are forcing 1-step intervals
        if self.forced_interval:
            info = self._take_no_additional_steps(info)
            info['done'][-1] = done
            # return obs, reward, (term or trunc), False, info
            return obs, reward, term, trunc, info

        # Take 1-step only (or if environment has already terminated)
        if done or self.current_step_mode == 0:
            # Update step_mode for next observation
            self._flip_step_modes()

            info = self._take_no_additional_steps(info)
            info['done'][-1] = done
            # return obs, reward, (term or trunc), False, info
            return obs, reward, term, trunc, info

        # Take 5-steps (additional 4 steps)
        elif self.current_step_mode == 1:
            gamma = self.gamma
            # Update step_mode for next observation
            self._flip_step_modes()

            for _ in range(4):
                # Take another step (second)
                obs, reward, term, trunc, info, gamma = self._take_another_step(action, reward, gamma, info)
                done = term or trunc
                if done:
                    break

            info['done'][-1] = done
            return obs, reward, term, trunc, info

        else:
            raise ValueError(f"Invalid step_mode: {self.step_mode}")

    def _flip_step_modes(self):
        self.last_step_mode = self.current_step_mode
        self.current_step_mode = np.random.randint(0, 2)

    def _take_first_step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, trunc, info = self.env.step(action)
        info['obs'] = [obs]
        info['action'] = [action]
        info['reward'] = [reward]
        info['done'] = [term or trunc]
        info['bonus_steps_taken'] = 0
        return obs, reward, term, trunc, info

    @staticmethod
    def _take_no_additional_steps(info):
        return info

    def _take_another_step(self, action: Any, reward: Any, gamma: float, base_info: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Take additional step
        base_info['bonus_steps_taken'] += 1

        obs, new_reward, term, trunc, new_info = self.env.step(action)
        reward += gamma * new_reward

        # Update return variables
        base_info = self._update_info(base_info, new_info)
        base_info['obs'].append(obs)
        base_info['action'].append(action)
        base_info['reward'].append(new_reward)
        base_info['done'].append(term or trunc)
        return obs, reward, term, trunc, base_info, gamma * self.gamma

    @staticmethod
    def _update_info(base_info, new_info):
        base_info.update(new_info)
        return base_info


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
        low, high = np.concatenate([[0], low]), np.concatenate([[1], high])
        self.state_dim = (self.state_dim[0] + 1,)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=self.state_dim, dtype=self.dtype
        )
        self.use_flag = use_flag

    def observation(self, obs):
        # Concat flag (0/1) to the start of the channels
        # - always set to 0 if use_flag = False
        val = 1 if self.env.get_wrapper_attr("current_step_mode") == 1 and self.use_flag else 0 # 0 for no repeat, 1 for repeat
        flag = np.full((1,), val, dtype=np.uint8)
        return np.concatenate([flag, obs], axis=-1)


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
            # Always set decoy obs flag to zero and concat to the start of the channels
            flag = np.full((1,), 0, dtype=vanilla_obs.dtype)
            obs = np.concatenate([flag, vanilla_obs], axis=-1)
            info['obs'][idx] = obs
        return info


def make_glucose_env(*, no_interim_rewards: bool = True, gamma: float = 1.0, forced_interval: int = 0,
                     use_flag: bool = True, **kwargs):
    env = T1DPatientEnv(**kwargs)
    env = CombinedObservationWrapper(env)
    if no_interim_rewards:
        env = EpisodeRewardsOnly(env)
    env = NormalizeObservation(env)
    env = AlternateStepWrapper(env, forced_interval=forced_interval, gamma=gamma)
    env = RepeatFlagChannel(env, use_flag=use_flag)     # +1 channel flag
    env = DecoyObsWrapper(env)
    return env
