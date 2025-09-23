import os

from gymnasium.envs.registration import register, WrapperSpec
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import pickle
import argparse
from collections import defaultdict
import polars as pl

from utils import *
from gym_wrappers import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--train_ppo', default=False, type=parse_bool, help='Train PPO agent')
parser.add_argument('--train_iql', default=False, type=parse_bool, help='Train IQL agent')
parser.add_argument('--ppo_agent', default=None, type=str, help='Path to pre-trained PPO agent')
parser.add_argument('--render_performance', default=False, type=parse_bool, help='Whether to render performance in final eval')
parser.add_argument('--record_video', default=False, type=parse_bool, help='Whether to record video of performance rendering')

parser.add_argument('--expectile', default=0.5, type=float, help='Expectile value for IQL training (0.5 is BC)')
parser.add_argument('--dropout_p', default=0.0, type=float, help='MC dropout probability for PPO agent')
parser.add_argument('--decoy_interval', default=0, type=int, help='Decoy interval: 0 (natural), 1 (1-step), 2 (2-step)')

GAMMA = 0.99


"""
Trained on natural dataset:
- evaluated on natural environment
- evaluated on forced 1-step environment - flag forced to 0

Trained on artificial 2-step decoy dataset - flag forced to 0
- evaluated on natural environment - flag forced to 0
- evaluated on forced 1-step environment - flag forced to 0

Trained on artificial 1-step decoy dataset - flag forced to 0
- evaluated on natural environment - flag forced to 0
- evaluated on forced 1-step environment - flag forced to 0
"""

additional_wrappers = (
    WrapperSpec(
        name="RecordVideo",
        entry_point="gymnasium.wrappers.record_video:RecordVideo",
        kwargs={
            'video_folder': current_video_folder,
            'episode_trigger': all_episodes_trigger,
            'step_trigger': None,
            'video_length': 0,
            'name_prefix': 'rl-video',
            'disable_logger': False
        },
    ),
    # For debugging purposes only
    # WrapperSpec(
        # name="FullyObsWrapper",
        # entry_point="minigrid.wrappers:FullyObsWrapper",
        # kwargs=None,
    # ),
    WrapperSpec(
        name="AlternateStepWrapper",
        entry_point="gym_wrappers:AlternateStepWrapper",
        kwargs={},
    ),
    WrapperSpec(
        name="RecordableImgObsWrapper",
        entry_point="gym_wrappers:RecordableImgObsWrapper",
        kwargs={},
    ),
    WrapperSpec(
        name="RepeatFlagChannel",
        entry_point="gym_wrappers:RepeatFlagChannel",
        kwargs={},
    ),
    WrapperSpec(
        name="DecoyObsWrapper",
        entry_point="gym_wrappers:DecoyObsWrapper",
        kwargs={},
    ),
)
no_video_additional_wrappers = additional_wrappers[1:]


register(
    id="LavaGapS6AltStep-v0",
    entry_point="gym_wrappers:make_lavastep_env",
    additional_wrappers=no_video_additional_wrappers,
)
register(
    id="LavaGapS6AltStepWithVideo-v0",
    entry_point="gym_wrappers:make_video_lavastep_env",
    additional_wrappers=additional_wrappers,
)


if __name__ == "__main__":
    args = parser.parse_args()

    train_ppo = args.train_ppo
    train_iql = args.train_iql
    render_performance = args.render_performance
    record_video = args.record_video

    if (train_iql and not os.path.exists('./dataset.pkl')) or render_performance:
        ppo_agent = f'../logs/ppo_minigrid_logs/{args.ppo_agent}' if args.ppo_agent is not None else None
        if ppo_agent is None:
            ppo_agent = choose_ppo_agent()
        assert ppo_agent is not None, ("Please provide a pre-trained PPO agent from the logs/ppo_minigrid_logs folder "
                                       "for IQL training.")
        assert os.path.exists(ppo_agent), "Provided PPO agent path does not exist."

    assert not (train_ppo and train_iql), "Please choose to train either PPO or IQL, not both."

    EXPECTILE = args.expectile
    DECOY_INTERVAL = args.decoy_interval

    model_loaded = False

    policy_kwargs = dict(
        features_extractor_class=PPOMiniGridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    env_name = "LavaGapS6AltStep-v0"
    video_env_name = "LavaGapS6AltStepWithVideo-v0"
    if train_ppo:
        # Create eval callback
        save_each_best = SaveEachBestCallback(save_dir="../logs/ppo_minigrid_logs/historic_bests", verbose=1)
        eval_callback = EvalCallback(gym.make(env_name, max_steps=100, fixed_reward=False),
                                     n_eval_episodes=100,
                                     callback_on_new_best=CallbackList([save_each_best]),
                                     verbose=1,
                                     eval_freq=2000,
                                     deterministic=False,
                                     best_model_save_path="../logs/ppo_minigrid_logs")

        model = PPO("CnnPolicy", gym.make(env_name, max_steps=100), ent_coef=0.1,
                    policy_kwargs=policy_kwargs, gamma=GAMMA, verbose=1)
        model.learn(2e5, callback=eval_callback)  # Train for 500,000 step with early stopping
        model_loaded = True

    if train_iql:
        print(f"EXPECTILE: {EXPECTILE}, DECOY_INTERVAL: {DECOY_INTERVAL}")

        logs = defaultdict(list)

        # Get our PPO model
        base_env = gym.make(env_name, max_steps=50)

        # Fill our replay buffer (or load pre-filled)
        replay_buffer_env = ReplayBufferEnv(base_env, buffer_size=1000000)
        if not os.path.exists('./dataset.pkl'):
            model = CallablePPO.load(ppo_agent, env=base_env, device="auto")
            model_loaded = True
            replay_buffer_env.fill_buffer(model=model, n_frames=100_000)
            with open('./dataset.pkl', 'wb') as f:
                pickle.dump(replay_buffer_env, f)
                f.close()
        else:
            print('='*50, '\nRe-using existing dataset.pkl...\n', '='*50)
            with open('./dataset.pkl', 'rb') as f:
                replay_buffer_env = pickle.load(f)
                f.close()

        dataset_rewards = np.array(replay_buffer_env.rewards[0])[np.array(replay_buffer_env.dones[0]) == 1]
        dataset_n_episodes = len(dataset_rewards)

        print(f"Baseline reward of the dataset: {dataset_rewards.mean():.2f} "
              f"+/- {dataset_rewards.std() / np.sqrt(dataset_n_episodes):.2f}")

        # Get our evaluators
        evaluators = {}
        for key, (interval, flag) in [
            ["lavagap_1_3", (0, not DECOY_INTERVAL)],
            ["lavagap_1_1", (1, False)],
        ]:
            evaluators[key] = EnvironmentEvaluator(gym.make(env_name,
                                                            max_steps=50,
                                                            use_flag=flag,
                                                            forced_interval=interval),
                                                   n_trials=500)

        for n_trial in range(10):
            logs['expectile'].append(EXPECTILE)
            logs['decoy_interval'].append(DECOY_INTERVAL)
            logs['dataset_reward'].append(dataset_rewards.mean())

            # Alternately collect and training
            algo = CustomIQL(observation_shape=base_env.observation_space.shape,
                             action_size=base_env.action_space.n,
                             feature_size=32,
                             batch_size=32,
                             expectile=EXPECTILE,
                             gamma=GAMMA,
                             dropout_p=args.dropout_p,
                             device='cuda' if torch.cuda.is_available() else 'cpu')

            algo.compile()

            log_dict = algo.fit(
                dataset=replay_buffer_env,
                epochs=1,
                n_steps_per_epoch=10_000,
                evaluators=evaluators,
                dataset_kwargs={'decoy_interval': DECOY_INTERVAL},
            )
            for key in evaluators.keys():
                logs[f'{key}_eval'].append(log_dict[key][0])

        # Save logs
        os.makedirs('../logs/iql_minigrid_logs', exist_ok=True)
        pl.DataFrame(logs).write_csv(f'../logs/iql_minigrid_logs/log_expectile={EXPECTILE}_decoy={DECOY_INTERVAL}.csv')

    if render_performance:
        eval_env = gym.make(video_env_name if record_video else env_name,
                            render_mode="rgb_array" if record_video else "human",
                            max_steps=100,
                            tile_size=128)

        if not model_loaded:
            model = CallablePPO.load(ppo_agent, env=eval_env, device="auto")

        total_episodes = 10
        for ep_number in range(total_episodes):
            observation, info = eval_env.reset(seed=42+ep_number)
            done = False
            while not done:
                action = model.predict(observation)[0]
                observation, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
        eval_env.close()