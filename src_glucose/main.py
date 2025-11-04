from collections import defaultdict
from functools import partial

import polars as pl
from scipy.stats import trim_mean

from gym_wrappers import *
from models import *
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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    if train_offline:
        if ppo_agent is None:
            ppo_agent = choose_ppo_agent()
        assert ppo_agent is not None, (
            "Please provide a pre-trained PPO agent from the logs_glucose/ppo_logs folder "
            "for IQL training.")
        assert os.path.exists(ppo_agent), "Provided PPO agent path does not exist."

    assert not (train_ppo and train_offline), "Please choose to train either PPO or IQL, not both."

    EXPECTILE = args.expectile
    DECOY_INTERVAL = args.decoy_interval

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
        ppo_agent = RecurrentPPO.load_checkpoint(ppo_agent, base_env)

        # Fill our replay buffer (or load pre-filled)
        dataset_size = 10_000_000
        dataset = RecurrentReplayBufferEnv(base_env, buffer_size=dataset_size * 10)
        if not os.path.exists('./replay_buffer/COMPLETE'):
            dataset.fill_buffer(model=ppo_agent, n_frames=dataset_size)
            dataset.save('./replay_buffer')
        else:
            print('=' * 50, '\nRe-using existing dataset...\n', '=' * 50)
            dataset.load('./replay_buffer')

        dataset_sem = dataset.dataset_IQR_std / np.sqrt(dataset.dataset_IQR_n_episodes)
        mean_ep_duration = np.array(dataset.observations[0]).shape[0] / sum(dataset.dones[0])
        print(f"\n=================\nBaseline IQR return of the dataset: "
              f"{dataset.dataset_IQR_return:.2f}"
              f"+/- {dataset_sem:.2f}\nMean duration: {int(mean_ep_duration)} steps\n=================\n")

        # Get our evaluators
        evaluators = dict()
        # - OPE evaluation
        """
        evaluators["wis_ope"] = WISOPEEvaluator(ppo_agent=ppo_agent,
                                                dataset=train_dataset,
                                                gamma=GAMMA,
                                                decoy_interval=DECOY_INTERVAL,
                                                device=DEVICE)
        """
        # - Online evaluation
        for key, interval in [
            ["online_irregular", 0],
            ["online_interpolated", 1],
        ]:
            evaluators[key] = ParallelEnvironmentEvaluator(partial(make_glucose_env,
                                                                   forced_interval=interval,
                                                                   use_test_ids=True),
                                                           n_eval_envs=100,
                                                           n_eval_episodes=100,
                                                           gamma=GAMMA,
                                                           verbose=False)

        if DECOY_INTERVAL == 2:
            evaluators["online_irregular_aggregated"] = ParallelEnvironmentEvaluator(partial(make_glucose_env,
                                                                                             forced_interval=0,
                                                                                             use_test_ids=True),
                                                                                     n_eval_envs=100,
                                                                                     n_eval_episodes=100,
                                                                                     gamma=GAMMA,
                                                                                     verbose=False)

        # 10 independent runs of the experiment
        all_scores = defaultdict(list)
        n_runs = 30
        # Set training params
        epoch_frac = 1.0
        if DECOY_INTERVAL in [0, 1]:
            n_train_epochs = 10
            n_epochs_per_eval = 1
        elif DECOY_INTERVAL == 2:
            n_train_epochs = 1000
            n_epochs_per_eval = 5
        else:
            raise ValueError("Invalid decoy interval.")

        # Log meta data
        algo = 'bc' if is_iql and EXPECTILE == 0.5 else args.offline_model
        model_save_path = f"../logs_glucose/iql_minigrid_models/{algo}"
        logs['algo'].append(algo)
        logs['decoy_interval'].append(DECOY_INTERVAL)
        logs['dataset_iqr_return'].append(dataset.dataset_IQR_return)
        logs['dataset_iqr_std'].append(dataset.dataset_IQR_std)
        for seed in np.arange(n_runs) * 1000:

            # Initialise offline model
            algo = offline_model(observation_shape=base_env.observation_space.shape,
                                 action_space=base_env.action_space,
                                 hidden_dim=128,
                                 recurrent_hidden_size=128,
                                 batch_size=1024,
                                 expectile=EXPECTILE,
                                 gamma=GAMMA,
                                 value_lr=3e-4,
                                 policy_lr=3e-4,
                                 critic_lr=3e-4,
                                 beta=args.beta,
                                 seed=seed,
                                 device=DEVICE)

            log_dict = algo.fit(
                dataset=dataset,
                n_epochs_train=n_train_epochs,
                n_epochs_per_eval=n_epochs_per_eval,
                evaluators=evaluators,
                decoy_interval=DECOY_INTERVAL,
                dataset_kwargs={'epoch_fraction': epoch_frac},
            )
            for key in evaluators.keys():
                all_scores[key].append(log_dict[key])

            # Save model state dicts in pytorch
            current_model_path = os.path.join(model_save_path, f'seed={seed}.pt')
            os.makedirs(model_save_path, exist_ok=True)
            algo.save_checkpoint(current_model_path)

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
        os.makedirs('../logs_glucose/iql_minigrid_logs', exist_ok=True)
        if is_iql:
            csv_path = (f'../logs_glucose/iql_minigrid_logs/decoy={DECOY_INTERVAL}_expectile={EXPECTILE}'
                        f'_beta={args.beta}.csv')
        elif is_cql:
            csv_path = f'../logs_glucose/iql_minigrid_logs/decoy={DECOY_INTERVAL}_CQL.csv'
        else:
            raise ValueError("Invalid offline model type.")
        pl.DataFrame(logs).write_csv(csv_path)
