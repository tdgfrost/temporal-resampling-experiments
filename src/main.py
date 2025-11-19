import os

from gymnasium.envs.registration import register, WrapperSpec
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import pickle
import argparse
from math import ceil
from functools import partial
from collections import defaultdict
import polars as pl
from scipy.stats import trim_mean

from utils import *
from gym_wrappers import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_ppo', default=False, type=parse_bool, help='Train PPO agent')
parser.add_argument('--train_offline', default=False, type=parse_bool, help='Train IQL agent')
parser.add_argument('--offline_model', default='iql', type=str, choices=['iql', 'cql', 'ppo'],
                    help='Type of offline RL model to train (iql or cql)')
parser.add_argument('--ppo_agent', default=None, type=str, help='Path to pre-trained PPO agent')
parser.add_argument('--render_performance', default=False, type=parse_bool,
                    help='Whether to render performance in final eval')
parser.add_argument('--record_video', default=False, type=parse_bool,
                    help='Whether to record video of performance rendering')

parser.add_argument('--alpha', default=1.0, type=float, help='Alpha parameter for CQL (1.0 is default)')
parser.add_argument('--expectile', default=0.8, type=float, help='Expectile value for IQL training (0.5 is BC)')
parser.add_argument('--dropout_p', default=0.0, type=float, help='MC dropout probability for PPO agent')
parser.add_argument('--beta', default=10.0, type=float, help='Beta parameter for IQL agent')
parser.add_argument('--decoy_interval', default=0, type=int, help='Decoy interval: 0 (natural), 1 (1-step), 2 (2-step)')

GAMMA = 0.99
MASTER_SEED = 123

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
    train_offline = args.train_offline
    is_iql = args.offline_model == 'iql'
    is_cql = args.offline_model == 'cql'
    is_ppo = args.offline_model == 'ppo'
    assert not train_offline or is_iql or is_cql or is_ppo, (
        "Please choose a valid offline model: 'iql', 'cql', or 'ppo'.")
    offline_model = CustomIQL if is_iql else (CustomCQLSAC if is_cql else None)
    render_performance = args.render_performance
    record_video = args.record_video

    if (train_offline and not os.path.exists('./dataset.pkl')) or (train_offline and is_ppo) or render_performance:
        ppo_agent = f'../logs/ppo_minigrid_logs/{args.ppo_agent}' if args.ppo_agent is not None else None
        if ppo_agent is None:
            ppo_agent = choose_ppo_agent()
        assert ppo_agent is not None, "Please provide a pre-trained PPO agent from the logs/ppo_minigrid_logs folder."
        assert os.path.exists(ppo_agent), "Provided PPO agent path does not exist."

    assert not (train_ppo and train_offline), "Please choose to train either PPO or IQL, not both."

    ALPHA = args.alpha
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
        eval_callback = EvalCallback(gym.make(env_name, max_steps=100, use_flag=False),
                                     n_eval_episodes=100,
                                     callback_on_new_best=CallbackList([save_each_best]),
                                     verbose=1,
                                     eval_freq=2000,
                                     deterministic=False,
                                     best_model_save_path="../logs/ppo_minigrid_logs")

        model = PPO("CnnPolicy", gym.make(env_name, max_steps=100, use_flag=False, fixed_reward=False), ent_coef=0.1,
                    policy_kwargs=policy_kwargs, gamma=GAMMA, verbose=1)
        model.learn(2e5, callback=eval_callback)  # Train for 500,000 step with early stopping
        model_loaded = True

    if train_offline:
        if is_iql:
            print(
                f"DECOY_INTERVAL: {DECOY_INTERVAL}, EXPECTILE: {EXPECTILE}, BETA: {args.beta}")
        elif is_cql:
            print(
                f"DECOY_INTERVAL: {DECOY_INTERVAL}, ALPHA: {ALPHA}")
        elif is_ppo:
            print("No offline training - PPO evaluation only.")

        logs = defaultdict(list)
        all_scores = defaultdict(list)

        algo_name = 'bc' if is_iql and EXPECTILE == 0.5 else args.offline_model
        logs['algo'].append(algo_name)
        logs['decoy_interval'].append(DECOY_INTERVAL)

        # Create our random seeds
        rng = np.random.default_rng(MASTER_SEED)
        n_runs = 50
        experiment_seeds = rng.integers(low=0, high=2**32 - 1, size=n_runs)

        # Get our evaluators
        evaluators = {}
        for key, (interval, flag) in [
            ["lavagap_1_3_eval", (0, not DECOY_INTERVAL)],
            ["lavagap_1_1_eval", (1, False)],
        ]:
            evaluators[key] = ParallelEnvironmentEvaluator(partial(gym.make,
                                                                   env_name,
                                                                   max_steps=50,
                                                                   use_flag=False,  # flag,
                                                                   forced_interval=interval),
                                                           n_eval_episodes=100,
                                                           n_eval_envs=20)

        # Get our PPO model
        base_env = gym.make(env_name, max_steps=50)

        # Do PPO evaluation here
        if is_ppo:
            # Load PPO model
            model = CallablePPO.load(ppo_agent, env=base_env, device="auto")
            for seed in tqdm(experiment_seeds, mininterval=2, desc="PPO evaluation"):
                for key in evaluators.keys():
                    all_scores[key].append(evaluators[key](model, seed=int(seed))[0])

        else:
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
                print('=' * 50, '\nRe-using existing dataset.pkl...\n', '=' * 50)
                with open('./dataset.pkl', 'rb') as f:
                    replay_buffer_env = pickle.load(f)
                    f.close()

            dataset_rewards = np.array(replay_buffer_env.rewards[0])[np.array(replay_buffer_env.dones[0]) == 1]
            dataset_n_episodes = len(dataset_rewards)

            print(f"Baseline reward of the dataset: {dataset_rewards.mean():.2f} "
                  f"+/- {dataset_rewards.std() / np.sqrt(dataset_n_episodes):.2f}")

            for seed_idx, seed in enumerate(experiment_seeds):
                # Alternately collect and training
                algo = offline_model(observation_shape=base_env.observation_space.shape,
                                     action_size=base_env.action_space.n,
                                     feature_size=ceil(64 / (1 - args.dropout_p)),
                                     batch_size=ceil(64 / (1 - args.dropout_p)),
                                     expectile=EXPECTILE,
                                     gamma=GAMMA,
                                     dropout_p=args.dropout_p,
                                     beta=args.beta,
                                     cql_alpha=args.alpha,
                                     critic_lr=1e-3,
                                     value_lr=1e-3,
                                     actor_lr=1e-3,
                                     seed=seed,
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
                    all_scores[key].append(log_dict[key][0])

        # Calculate our bootstrap IQM for each evaluator
        for key, scores in all_scores.items():
            scores = np.array(scores)
            bootstrap_scores = np.random.choice(scores, size=(100_000, n_runs), replace=True)
            bootstrap_iqm = trim_mean(bootstrap_scores, proportiontocut=0.25, axis=-1)
            bootstrap_low_q = np.percentile(bootstrap_iqm, 2.5)
            bootstrap_high_q = np.percentile(bootstrap_iqm, 97.5)
            logs[f'{key}_iqm'] = np.mean(bootstrap_iqm)
            logs[f'{key}_2.5%'] = bootstrap_low_q
            logs[f'{key}_97.5%'] = bootstrap_high_q

        # Save logs
        os.makedirs('../logs/iql_minigrid_logs', exist_ok=True)
        if is_iql:
            csv_path = (f'../logs/iql_minigrid_logs/decoy={DECOY_INTERVAL}_expectile={EXPECTILE}'
                        f'_beta={args.beta}_{algo_name}.csv')
        elif is_cql:
            csv_path = f'../logs/iql_minigrid_logs/decoy={DECOY_INTERVAL}_alpha={ALPHA}_cql.csv'
        elif is_ppo:
            csv_path = f"../logs/iql_minigrid_logs/ppo_baseline.csv"
        pl.DataFrame(logs).write_csv(csv_path)

        # Close the eval environments
        for key in evaluators.keys():
            evaluators[key].vec_env.close()

    if render_performance:
        eval_env = gym.make(video_env_name if record_video else env_name,
                            render_mode="rgb_array" if record_video else "human",
                            max_steps=100,
                            tile_size=128)

        if not model_loaded:
            model = CallablePPO.load(ppo_agent, env=eval_env, device="auto")

        total_episodes = 10
        for ep_number in range(total_episodes):
            observation, info = eval_env.reset(seed=42 + ep_number)
            done = False
            while not done:
                action = model.predict(observation)[0]
                observation, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
        eval_env.close()
