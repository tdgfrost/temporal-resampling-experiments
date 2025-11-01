import polars as pl
from collections import defaultdict
from functools import partial
from scipy.stats import trim_mean

from gym_wrappers import *
from models import *
from ppo_trainer import RecurrentPPO
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_ppo', default=False, type=parse_bool, help='Train PPO agent')
parser.add_argument('--train_offline', default=False, type=parse_bool, help='Train offline agent')
parser.add_argument('--offline_model', default='iql', type=str, choices=['iql', 'cql'],
                    help='Type of offline RL model to train (iql or cql)')
parser.add_argument('--ppo_agent', default="best_model17_6837.02.pth", type=str, help='Path to pre-trained PPO agent')

parser.add_argument('--expectile', default=0.9, type=float, help='Expectile value for IQL training (0.5 is BC)')
parser.add_argument('--beta', default=10., type=float, help='Beta parameter for IQL agent')
parser.add_argument('--decoy_interval', default=0, type=int, help='Decoy interval: 0 (natural), 1 (1-step), 2 (2-step)')

GAMMA = 0.99

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    args = parser.parse_args()

    train_ppo = args.train_ppo
    train_offline = args.train_offline
    is_iql = args.offline_model == 'iql'
    is_cql = args.offline_model == 'cql'
    ppo_agent = f'../logs_glucose/ppo_logs/{args.ppo_agent}' if args.ppo_agent is not None else None
    assert not train_offline or is_iql or is_cql, (
        "Please choose a valid offline model: 'iql' or 'cql'.")
    offline_model = RecurrentIQL if is_iql else (RecurrentCQLSAC if is_cql else None)

    if train_offline and not os.path.exists('./replay_buffer/COMPLETE'):
        if ppo_agent is None:
            ppo_agent = choose_ppo_agent()
        assert ppo_agent is not None, (
            "Please provide a pre-trained PPO agent from the logs_glucose/ppo_logs folder "
            "for IQL training.")
        assert os.path.exists(ppo_agent), "Provided PPO agent path does not exist."

    assert not (train_ppo and train_offline), "Please choose to train either PPO or IQL, not both."

    EXPECTILE = args.expectile
    DECOY_INTERVAL = args.decoy_interval

    model_loaded = False

    if train_ppo:
        GAMMA = 0.99

        base_env = make_glucose_env(use_scaling=True)
        env = EnforcePPOWrapper(base_env, gamma=GAMMA)
        env_creator_fn = partial(make_glucose_env, use_test_ids=True)

        # *** KEY CHANGE: UPDATED HYPERPARAMETERS ***
        agent = RecurrentPPO(env, env_creator_fn=env_creator_fn, gamma=GAMMA,
                             n_steps=1028,  # More data per update
                             entropy_coef=0.01,  # Too high = too unstable
                             clip_range=0.2,  # Relax the clip range
                             batch_size=256,
                             gae_lambda=0.95,
                             n_epochs=5,  # Fewer epochs
                             hidden_dim=128,
                             seed=123,
                             learning_rate=3e-4,  # Standard learning rate
                             eval_freq=100_000,
                             eval_episodes=500)
        agent.fit(total_timesteps=10_000_000)

    if train_offline:
        if is_iql:
            print(
                f"DECOY_INTERVAL: {DECOY_INTERVAL}, EXPECTILE: {EXPECTILE}, BETA: {args.beta}")
        else:
            print(
                f"DECOY_INTERVAL: {DECOY_INTERVAL}")

        logs = defaultdict(list)

        # Get our PPO model
        base_env = make_glucose_env()

        # Fill our replay buffer (or load pre-filled)
        dataset_size = 1_000_000
        replay_buffer_env = RecurrentReplayBufferEnv(base_env, buffer_size=dataset_size * 10)
        if not os.path.exists('./replay_buffer/COMPLETE'):
            model = RecurrentPPO.load_checkpoint(ppo_agent, base_env)
            model_loaded = True
            replay_buffer_env.fill_buffer(model=model, n_frames=dataset_size)
            replay_buffer_env.save('./replay_buffer')
        else:
            print('=' * 50, '\nRe-using existing dataset...\n', '=' * 50)
            replay_buffer_env.load('./replay_buffer')

        dataset_sem = replay_buffer_env.dataset_IQR_std / np.sqrt(replay_buffer_env.dataset_IQR_n_episodes)
        mean_ep_duration = np.array(replay_buffer_env.observations[0]).shape[0] / sum(replay_buffer_env.dones[0])
        print(f"\n=================\nBaseline IQR return of the dataset: "
              f"{replay_buffer_env.dataset_IQR_return:.2f}"
              f"+/- {dataset_sem:.2f}\nMean duration: {int(mean_ep_duration)} steps\n=================\n")

        # 10 independent runs of the experiment
        all_scores = defaultdict(list)
        n_runs = 10
        # Log meta data
        algo = 'bc' if is_iql and EXPECTILE == 0.5 else args.offline_model
        logs['algo'].append(algo)
        logs['decoy_interval'].append(DECOY_INTERVAL)
        logs['dataset_iqr_return'].append(replay_buffer_env.dataset_IQR_return)
        logs['dataset_iqr_std'].append(replay_buffer_env.dataset_IQR_std)
        for seed in np.arange(n_runs) * 1000:
            # Get our evaluators
            evaluators = {}
            for key, interval in [
                ["glucose_irregular", 0],
                ["glucose_regular", 1],
            ]:
                evaluators[key] = ParallelEnvironmentEvaluator(partial(make_glucose_env,
                                                                       forced_interval=interval,
                                                                       use_test_ids=True),
                                                               n_eval_envs=5,
                                                               n_eval_episodes=100)

            if DECOY_INTERVAL == 2:
                evaluators["glucose_irregular_aggregated"] = ParallelEnvironmentEvaluator(partial(make_glucose_env,
                                                                                                  forced_interval=0,
                                                                                                  use_test_ids=True),
                                                                                          n_eval_envs=5,
                                                                                          n_eval_episodes=100)

            # Alternately collect and training
            algo = offline_model(observation_shape=base_env.observation_space.shape,
                                 action_space=base_env.action_space,
                                 hidden_dim=128,
                                 batch_size=1024,
                                 expectile=EXPECTILE,
                                 gamma=GAMMA,
                                 value_lr=3e-4,
                                 policy_lr=3e-4,
                                 critic_lr=3e-4,
                                 beta=args.beta,
                                 seed=seed,
                                 device='cuda' if torch.cuda.is_available() else 'cpu')

            algo.compile()

            epoch_frac = 1.0
            if DECOY_INTERVAL in [0, 1]:
                n_train_epochs = 5
            elif DECOY_INTERVAL == 2:
                n_train_epochs = 1000
            else:
                raise ValueError("Invalid decoy interval.")

            log_dict = algo.fit(
                dataset=replay_buffer_env,
                n_epochs_train=n_train_epochs,
                n_epochs_per_eval=n_train_epochs,
                evaluators=evaluators,
                decoy_interval=DECOY_INTERVAL,
                dataset_kwargs={'epoch_fraction': epoch_frac},
            )
            for key in evaluators.keys():
                all_scores[key].append(log_dict[key])

        # Calculate our bootstrap IQM for each evaluator
        for key, scores in all_scores.items():
            scores = np.array(scores)
            bootstrap_scores = np.random.choice(scores, size=(100_000, 10), replace=True)
            bootstrap_iqm = trim_mean(bootstrap_scores, proportiontocut=0.25, axis=-1)
            bootstrap_low_q = np.percentile(bootstrap_iqm, 2.5)
            bootstrap_high_q = np.percentile(bootstrap_iqm, 97.5)
            logs[f'{key}_iqm'] = np.mean(bootstrap_iqm)
            logs[f'{key}_2.5%'] = bootstrap_low_q
            logs[f'{key}_97.5%'] = bootstrap_high_q

        # Save logs
        os.makedirs('../logs_glucose/iql_minigrid_logs', exist_ok=True)
        if is_iql:
            csv_path = (f'../logs_glucose/iql_minigrid_logs/decoy={DECOY_INTERVAL}_expectile={EXPECTILE}'
                        f'_beta={args.beta}.csv')
        elif is_cql:
            csv_path = f'../logs_glucose/iql_minigrid_logs/decoy={DECOY_INTERVAL}_CQL.csv'
        else:
            raise ValueError("Invalid offline model type.")
        pl.DataFrame(logs).write_csv(csv_path)
