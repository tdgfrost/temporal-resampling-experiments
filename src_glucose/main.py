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

GAMMA = 0.99

torch.set_float32_matmul_precision('high')


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

    if train_iql and not os.path.exists('./replay_buffer/COMPLETE'):
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
        features_extractor_kwargs=dict(features_dim=128),
        activation_fn=nn.ReLU,
        net_arch=[128, 128],
    )


    if train_ppo:
        # Create eval callback
        save_each_best = SaveEachBestCallback(save_dir="../logs_glucose/ppo_minigrid_logs/historic_bests", verbose=1)

        eval_callback = EvalCallback(make_glucose_env(),
                                     n_eval_episodes=20,
                                     callback_on_new_best=CallbackList([save_each_best]),
                                     verbose=1,
                                     eval_freq=2000,
                                     deterministic=True,
                                     best_model_save_path="../logs_glucose/ppo_minigrid_logs")

        model = RecurrentPPO(CustomRecurrentPolicy, env=make_glucose_env(), learning_rate=0.0003, ent_coef=0.01,
                             clip_range=0.2, policy_kwargs=policy_kwargs, gamma=1.0, verbose=1, device='cpu')
        model.learn(1e6, callback=eval_callback)  # Train for 500,000 step with early stopping
        model_loaded = True

    if train_iql:
        print(f"EXPECTILE: {EXPECTILE}, DECOY_INTERVAL: {DECOY_INTERVAL}, DROPOUT_P: {args.dropout_p}, BETA: {args.beta}")

        logs = defaultdict(list)

        # Get our PPO model
        base_env = make_glucose_env(no_interim_rewards=True)

        # Fill our replay buffer (or load pre-filled)
        dataset_size = 500_000
        replay_buffer_env = RecurrentReplayBufferEnv(base_env, buffer_size=dataset_size * 10)
        if not os.path.exists('./replay_buffer/COMPLETE'):
            model = CallableRecurrentPPO.load(ppo_agent, env=base_env, device="cpu")
            model_loaded = True
            replay_buffer_env.fill_buffer(model=model, n_frames=dataset_size)
            replay_buffer_env.save('./replay_buffer')
        else:
            print('='*50, '\nRe-using existing dataset...\n', '='*50)
            replay_buffer_env.load('./replay_buffer')

        print(f"Baseline reward of the dataset: {replay_buffer_env.dataset_avg:.2f} "
              f"+/- {replay_buffer_env.dataset_std:.2f}")

        # Get our evaluators
        evaluators = {}
        for key, (interval, flag) in [
            ["glucose_irregular", (0, not DECOY_INTERVAL)],
            ["glucose_regular", (1, False)],
        ]:
            evaluators[key] = EnvironmentEvaluator(make_glucose_env(use_flag=False, # flag,
                                                                    forced_interval=interval,
                                                                    no_interim_rewards=True),
                                                   n_trials=20,
                                                   min_scale_rewards = replay_buffer_env.min_rewards_scale,
                                                   max_scale_rewards = replay_buffer_env.max_rewards_scale,
            )

        if DECOY_INTERVAL == 2:
            evaluators["glucose_irregular_aggregated"] = EnvironmentEvaluator(make_glucose_env(use_flag=False, # flag,
                                                                                               forced_interval=interval,
                                                                                               no_interim_rewards=True),
                                                                              n_trials=20,
                                                                              running_average_obs=True,
                                                                              min_scale_rewards = replay_buffer_env.min_rewards_scale,
                                                                              max_scale_rewards = replay_buffer_env.max_rewards_scale,
                                                                              )

        for n_trial in range(10):
            logs['expectile'].append(EXPECTILE)
            logs['decoy_interval'].append(DECOY_INTERVAL)
            logs['dataset_reward'].append(replay_buffer_env.dataset_avg)

            # Alternately collect and training
            algo = RecurrentIQL(observation_shape=base_env.observation_space.shape,
                             action_space=base_env.action_space,
                             feature_size=ceil(128 / (1-args.dropout_p)),
                             batch_size=32,
                             expectile=EXPECTILE,
                             gamma=GAMMA,
                             decoy_interval=DECOY_INTERVAL,
                             dropout_p=args.dropout_p,
                             beta=args.beta,
                             device='cuda' if torch.cuda.is_available() else 'cpu')

            algo.compile()

            log_dict = algo.fit(
                dataset=replay_buffer_env,
                n_epochs_train=500 if DECOY_INTERVAL in [0, 2] else 5,
                n_epochs_eval=1,
                evaluators=evaluators,
                decoy_interval=DECOY_INTERVAL,
                dataset_kwargs={'decoy_interval': DECOY_INTERVAL, 'batch_size': 32},
            )
            for key in evaluators.keys():
                logs[f'{key}_eval'].append(log_dict[key][0])

        # Save logs
        os.makedirs('../logs_glucose/iql_minigrid_logs', exist_ok=True)
        pl.DataFrame(logs).write_csv(f'../logs_glucose/iql_minigrid_logs/log_expectile={EXPECTILE}_decoy={DECOY_INTERVAL}'
                                     f'_dropout={args.dropout_p}_beta={args.beta}.csv')
