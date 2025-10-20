from math import ceil
from collections import defaultdict
import polars as pl
from functools import partial

from utils import *
from gym_wrappers import *
from models import *
from ppo_trainer import RecurrentPPO

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
        ppo_agent = f'../logs_glucose/ppo_logs/{args.ppo_agent}' if args.ppo_agent is not None else None
        if ppo_agent is None:
            ppo_agent = choose_ppo_agent()
        assert ppo_agent is not None, (
            "Please provide a pre-trained PPO agent from the logs_glucose/ppo_logs folder "
            "for IQL training.")
        assert os.path.exists(ppo_agent), "Provided PPO agent path does not exist."

    assert not (train_ppo and train_iql), "Please choose to train either PPO or IQL, not both."

    EXPECTILE = args.expectile
    DECOY_INTERVAL = args.decoy_interval

    model_loaded = False

    if train_ppo:
        GAMMA = 0.9

        base_env = make_glucose_env()
        env = EnforcePPOWrapper(base_env, gamma=GAMMA)
        env_creator_fn = partial(make_glucose_env, use_test_ids=True)

        # *** KEY CHANGE: UPDATED HYPERPARAMETERS ***
        agent = RecurrentPPO(env, env_creator_fn=env_creator_fn, gamma=GAMMA,
                             n_steps=1028, # More data per update
                             entropy_coef=0.01,  # Can be slightly higher now
                             clip_range=0.2,  # Relax the clip range
                             batch_size=64,
                             gae_lambda=0.95,
                             n_epochs=10,  # Fewer epochs
                             hidden_dim=128,
                             seed=123,
                             learning_rate=1e-3,  # Standard learning rate
                             eval_freq=100_000,
                             eval_episodes=500,)
        agent.fit(total_timesteps=10_000_000)

    if train_iql:
        print(
            f"EXPECTILE: {EXPECTILE}, DECOY_INTERVAL: {DECOY_INTERVAL}, DROPOUT_P: {args.dropout_p}, BETA: {args.beta}")

        logs = defaultdict(list)

        # Get our PPO model
        base_env = make_glucose_env()

        # Fill our replay buffer (or load pre-filled)
        dataset_size = 100_000
        replay_buffer_env = RecurrentReplayBufferEnv(base_env, buffer_size=dataset_size * 10)
        if not os.path.exists('./replay_buffer/COMPLETE'):
            model = RecurrentPPO.load_checkpoint(ppo_agent, base_env)
            model_loaded = True
            replay_buffer_env.fill_buffer(model=model, n_frames=dataset_size)
            replay_buffer_env.save('./replay_buffer')
        else:
            print('=' * 50, '\nRe-using existing dataset...\n', '=' * 50)
            replay_buffer_env.load('./replay_buffer')

        dataset_rew_avg, dataset_rew_std = replay_buffer_env.dataset_avg, replay_buffer_env.dataset_std
        dataset_rew_min, dataset_rew_max = replay_buffer_env.min_rewards_scale, replay_buffer_env.max_rewards_scale
        dataset_rew_diff = dataset_rew_max - dataset_rew_min
        dataset_rew_avg_raw = dataset_rew_avg * dataset_rew_diff + dataset_rew_min
        mean_ep_duration = np.array(replay_buffer_env.observations[0]).shape[0] / sum(replay_buffer_env.dones[0])
        print(f"\n=================\nBaseline reward of the dataset: "
              f"{dataset_rew_avg:.2f} ({dataset_rew_avg_raw:.2f}) "
              f"+/- {dataset_rew_std:.2f}\nMean duration: {int(mean_ep_duration)} steps\n=================\n")

        # Get our evaluators
        evaluators = {}
        for key, (interval, flag) in [
            ["glucose_irregular", (0, not DECOY_INTERVAL)],
            ["glucose_regular", (1, False)],
        ]:
            evaluators[key] = EnvironmentEvaluator(make_glucose_env(use_flag=False,  # flag,
                                                                    forced_interval=interval,
                                                                    no_interim_rewards=True,
                                                                    use_test_ids=True),
                                                   n_trials=20,
                                                   min_scale_rewards=replay_buffer_env.min_rewards_scale,
                                                   max_scale_rewards=replay_buffer_env.max_rewards_scale,
                                                   )

        if DECOY_INTERVAL == 2:
            evaluators["glucose_irregular_aggregated"] = EnvironmentEvaluator(make_glucose_env(use_flag=False,  # flag,
                                                                                               forced_interval=interval,
                                                                                               no_interim_rewards=True,
                                                                                               use_test_ids=True),
                                                                              n_trials=20,
                                                                              running_average_obs=True,
                                                                              min_scale_rewards=replay_buffer_env.min_rewards_scale,
                                                                              max_scale_rewards=replay_buffer_env.max_rewards_scale,
                                                                              )

        for n_trial in range(10):
            logs['expectile'].append(EXPECTILE)
            logs['decoy_interval'].append(DECOY_INTERVAL)
            logs['dataset_reward'].append(replay_buffer_env.dataset_avg)

            # Alternately collect and training
            algo = RecurrentIQL(observation_shape=base_env.observation_space.shape,
                                action_space=base_env.action_space,
                                feature_size=ceil(128 / (1 - args.dropout_p)),
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
                n_epochs_train=500 if EXPECTILE == 0.5 else 200,
                n_epochs_eval=1,
                evaluators=evaluators,
                decoy_interval=DECOY_INTERVAL,
                dataset_kwargs={'decoy_interval': DECOY_INTERVAL, 'batch_size': 32},
            )
            for key in evaluators.keys():
                logs[f'{key}_eval'].append(log_dict[key][0])

        # Save logs
        os.makedirs('../logs_glucose/iql_minigrid_logs', exist_ok=True)
        pl.DataFrame(logs).write_csv(
            f'../logs_glucose/iql_minigrid_logs/log_expectile={EXPECTILE}_decoy={DECOY_INTERVAL}'
            f'_dropout={args.dropout_p}_beta={args.beta}.csv')
