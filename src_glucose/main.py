import os

from gymnasium.envs.registration import register, WrapperSpec
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import pickle
import argparse
from math import ceil
from collections import defaultdict
import polars as pl

from utils import *
from gym_wrappers import *
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--train_ppo', default=False, type=parse_bool, help='Train PPO agent')
parser.add_argument('--train_iql', default=False, type=parse_bool, help='Train IQL agent')
parser.add_argument('--ppo_agent', default=None, type=str, help='Path to pre-trained PPO agent')

parser.add_argument('--expectile', default=0.5, type=float, help='Expectile value for IQL training (0.5 is BC)')
parser.add_argument('--dropout_p', default=0.2, type=float, help='MC dropout probability for PPO agent')
parser.add_argument('--beta', default=1., type=float, help='Beta parameter for IQL agent')
parser.add_argument('--decoy_interval', default=0, type=int, help='Decoy interval: 0 (natural), 1 (1-step), 2 (2-step)')

GAMMA = 1.0


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


if __name__ == "__main__":
    args = parser.parse_args()
    if args.dropout_p > 0:
        args.beta = 0.

    train_ppo = args.train_ppo
    train_iql = args.train_iql

    if train_iql and not os.path.exists('./dataset.pkl'):
        ppo_agent = f'../logs_glucose/ppo_minigrid_logs/{args.ppo_agent}' if args.ppo_agent is not None else None
        if ppo_agent is None:
            ppo_agent = choose_ppo_agent()
        assert ppo_agent is not None, ("Please provide a pre-trained PPO agent from the logs_glucose/ppo_minigrid_logs folder "
                                       "for IQL training.")
        assert os.path.exists(ppo_agent), "Provided PPO agent path does not exist."

    assert not (train_ppo and train_iql), "Please choose to train either PPO or IQL, not both."

    EXPECTILE = args.expectile
    DECOY_INTERVAL = args.decoy_interval

    model_loaded = False

    policy_kwargs = dict(
        features_extractor_class=PPOMiniGridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=64),
    )

    if train_ppo:
        # Create eval callback
        save_each_best = SaveEachBestCallback(save_dir="../logs_glucose/ppo_minigrid_logs/historic_bests", verbose=1)
        # eval_callback = EvalCallback(make_sepsis_env(max_steps=100, fixed_reward=False),
        eval_callback = EvalCallback(make_glucose_env(),
                                     n_eval_episodes=10,
                                     callback_on_new_best=CallbackList([save_each_best]),
                                     verbose=1,
                                     eval_freq=2000,
                                     deterministic=False,
                                     best_model_save_path="../logs_glucose/ppo_minigrid_logs")

        model = PPO("MlpPolicy", make_glucose_env(),
                    ent_coef=0.01, policy_kwargs=policy_kwargs, gamma=GAMMA, verbose=1)
        model.learn(1e5, callback=eval_callback)  # Train for 500,000 step with early stopping
        model_loaded = True

    if train_iql:
        print(f"EXPECTILE: {EXPECTILE}, DECOY_INTERVAL: {DECOY_INTERVAL}, DROPOUT_P: {args.dropout_p}, BETA: {args.beta}")

        logs = defaultdict(list)

        # Get our PPO model
        base_env = make_glucose_env()

        # Fill our replay buffer (or load pre-filled)
        buffer_size = 100_000
        replay_buffer_env = ReplayBufferEnv(base_env, buffer_size=buffer_size)
        if not os.path.exists('./replay_buffer/COMPLETE'):
            model = CallablePPO.load(ppo_agent, env=base_env, device="auto")
            model_loaded = True
            replay_buffer_env.fill_buffer(model=model, n_frames=buffer_size)
            replay_buffer_env.save('./replay_buffer')
        else:
            print('='*50, '\nRe-using existing dataset.pkl...\n', '='*50)
            replay_buffer_env.load('./replay_buffer')

        dataset_rewards = np.array(replay_buffer_env.rewards[0])[np.array(replay_buffer_env.dones[0]) == 1]
        dataset_n_episodes = len(dataset_rewards)

        print(f"Baseline reward of the dataset: {dataset_rewards.mean():.2f} "
              f"+/- {dataset_rewards.std() / np.sqrt(dataset_n_episodes):.2f}")

        # Get our evaluators
        evaluators = {}
        for key, (interval, flag) in [
            ["glucose_1_3", (0, not DECOY_INTERVAL)],
            ["glucose_1_1", (1, False)],
        ]:
            evaluators[key] = EnvironmentEvaluator(make_glucose_env(use_flag=flag,
                                                                    forced_interval=interval),
                                                   n_trials=20,
                                                   min_scale_rewards=replay_buffer_env.min_rewards_scale,
                                                   max_scale_rewards=replay_buffer_env.max_rewards_scale)

        for n_trial in range(10):
            logs['expectile'].append(EXPECTILE)
            logs['decoy_interval'].append(DECOY_INTERVAL)
            logs['dataset_reward'].append(dataset_rewards.mean())

            # Alternately collect and training
            algo = CustomIQL(observation_shape=base_env.observation_space.shape,
                             action_space=base_env.action_space,
                             feature_size=ceil(128 / (1-args.dropout_p)),
                             batch_size=ceil(128 / (1-args.dropout_p)),
                             expectile=EXPECTILE,
                             gamma=GAMMA,
                             dropout_p=args.dropout_p,
                             beta=args.beta,
                             device='cuda' if torch.cuda.is_available() else 'cpu')

            algo.compile()

            log_dict = algo.fit(
                dataset=replay_buffer_env,
                epochs=1,
                n_steps_per_epoch=20_000,
                evaluators=evaluators,
                dataset_kwargs={'decoy_interval': DECOY_INTERVAL},
            )
            for key in evaluators.keys():
                logs[f'{key}_eval'].append(log_dict[key][0])

        # Save logs
        os.makedirs('../logs_glucose/iql_minigrid_logs', exist_ok=True)
        pl.DataFrame(logs).write_csv(f'../logs_glucose/iql_minigrid_logs/log_expectile={EXPECTILE}_decoy={DECOY_INTERVAL}'
                                     f'_dropout={args.dropout_p}_beta={args.beta}.csv')
