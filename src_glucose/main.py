from collections import defaultdict
from functools import partial

import polars as pl
from scipy.stats import trim_mean
from tqdm import tqdm

from gym_wrappers import *
from models import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_ppo', default=False, type=parse_bool, help='Train PPO agent')
parser.add_argument('--generate_dataset', default=False, type=parse_bool,
                    help='Generate dataset using pre-trained PPO agent')
parser.add_argument('--train_offline', default=False, type=parse_bool, help='Train offline agent')

parser.add_argument('--offline_model', default='iql', type=str, choices=['iql', 'cql', 'ppo'],
                    help='Type of offline RL model to train (iql or cql)')
parser.add_argument('--target_ppo_agent', default="best_model09_8144.51.pth", type=str,
                    help='Path to target PPO agent for dataset generation')

parser.add_argument('--expectile', default=0.9, type=float, help='Expectile value for IQL training (0.5 is BC)')
parser.add_argument('--beta', default=10., type=float, help='Beta parameter for IQL agent')
parser.add_argument('--decoy_interval', default=0, type=int, help='Decoy interval: 0 (natural), 1 (1-step), 2 (2-step)')

GAMMA = 0.99
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MASTER_SEED = 123

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    args = parser.parse_args()

    # Establish training mode/parameters
    train_ppo = args.train_ppo
    generate_dataset = args.generate_dataset
    train_offline = args.train_offline

    is_iql = args.offline_model == 'iql'
    is_cql = args.offline_model == 'cql'
    is_ppo = args.offline_model == 'ppo'  # Used for evaluation only

    assert sum([train_ppo, train_offline, generate_dataset]) == 1, \
        "Please select only one option (train_ppo, train_offline, generate_dataset)."

    assert not train_offline or is_iql or is_cql or is_ppo, (
        "Please choose a valid offline model: 'iql', 'cql', or 'ppo' (eval only).")

    if train_ppo:
        print(f'\n\n====== Training PPO ======\n\n')
        # Establish our PPO-specific ID setup
        train_ids = [1, 2, 3, 4, 5]
        test_ids = [6, 7, 8, 9, 10]
        GAMMA = 0.99

        base_env = make_glucose_env(use_scaling=True, test_ids=train_ids, use_test_ids=True)
        env = EnforcePPOWrapper(base_env, gamma=GAMMA)
        env_creator_fn = partial(make_glucose_env, test_ids=test_ids, use_test_ids=True)

        # *** KEY CHANGE: UPDATED HYPERPARAMETERS ***
        agent = RecurrentPPO(env, env_creator_fn=env_creator_fn, gamma=GAMMA,
                             n_steps=1024,  # More data per update
                             entropy_coef=0.01,  # Too high = too unstable
                             clip_range=0.2,  # Relax the clip range
                             batch_size=256,
                             gae_lambda=0.95,
                             n_epochs=5,  # Fewer epochs
                             hidden_dim=128,
                             seed=123,
                             learning_rate=3e-4,  # Standard learning rate
                             eval_freq=50_000,
                             eval_episodes=500)
        agent.fit(total_timesteps=10_000_000)

    elif generate_dataset or (train_offline and is_ppo):
        # Load our PPO agent for dataset generation / offline training
        ppo_agent = args.target_ppo_agent
        if ppo_agent is None:
            ppo_agent = choose_ppo_agent()
        ppo_agent = f'../logs_glucose/ppo_logs/{ppo_agent}'
        assert os.path.exists(ppo_agent), "Specified PPO agent path does not exist."
        dummy_env_creator = partial(make_glucose_env, test_ids=1)
        ppo_agent = RecurrentPPO.load_checkpoint(ppo_agent, dummy_env_creator(), env_creator_fn=dummy_env_creator)

    if generate_dataset:
        # Get our test ids
        for test_id in [6, 7, 8, 9, 10]:
            print(f'\n=== Generating dataset for test_id {test_id} ===\n')
            # Create our replay buffer
            dataset_size = 1_000_000
            dataset = RecurrentReplayBufferEnv(make_glucose_env(test_ids=test_id), buffer_size=dataset_size * 2)
            if not os.path.exists(f'./replay_buffers/replay_buffer_{test_id}/COMPLETE'):
                # Load PPO agent and fill buffer
                dataset.fill_buffer(model=ppo_agent, n_frames=dataset_size)
                dataset.save(f'./replay_buffers/replay_buffer_{test_id}')

    if train_offline:
        EXPECTILE = args.expectile
        DECOY_INTERVAL = args.decoy_interval
        algo = 'bc' if is_iql and EXPECTILE == 0.5 else args.offline_model

        if is_iql:
            print(f"\n=====\nDECOY_INTERVAL: {DECOY_INTERVAL}, EXPECTILE: {EXPECTILE}, BETA: {args.beta}\n=====\n")
        elif is_ppo:
            print('\n=====\nSpecial case - evaluating pre-trained PPO agent only.\n=====\n')
        elif is_cql:
            print(
                f"\n=====\nDECOY_INTERVAL: {DECOY_INTERVAL}\n=====\n")
        else:
            raise ValueError("Invalid offline model type.")

        logs = defaultdict(list)
        all_scores = defaultdict(list)

        logs['algo'].append(algo)
        if not is_ppo:
            logs['decoy_interval'].append(DECOY_INTERVAL)

        # Get our test ids
        for test_id in [6, 7, 8, 9, 10]:
            # Create our random seeds
            rng = np.random.default_rng(MASTER_SEED ** test_id)
            n_runs = 30
            experiment_seeds = rng.integers(low=0, high=2**32 - 1, size=n_runs)

            # Establish our test id setup
            make_glucose_env_custom = partial(make_glucose_env, test_ids=test_id)
            base_env = make_glucose_env_custom()

            # Get our evaluators
            evaluators = dict()

            # - Online evaluation
            for key, interval in [
                ["online_irregular", 0],
                ["online_interpolated", 1],
            ]:
                evaluators[key] = ParallelEnvironmentEvaluator(partial(make_glucose_env_custom,
                                                                       forced_interval=interval,
                                                                       use_test_ids=True),
                                                               n_eval_envs=100,
                                                               n_eval_episodes=200,
                                                               gamma=GAMMA,
                                                               verbose=False)

            if DECOY_INTERVAL == 2:
                evaluators["online_irregular_aggregated"] = ParallelEnvironmentEvaluator(partial(make_glucose_env_custom,
                                                                                                 forced_interval=0,
                                                                                                 use_test_ids=True),
                                                                                         n_eval_envs=100,
                                                                                         n_eval_episodes=200,
                                                                                         gamma=GAMMA,
                                                                                         verbose=False)

            if is_ppo:
                print(f'\n=== Evaluating pre-trained PPO for test_id {test_id} ===\n')
                with tqdm(total=n_runs) as pbar:
                    for seed in experiment_seeds:
                        eval_str = f"Test ID {test_id} PPO Evaluation Results:"
                        for key in evaluators.keys():
                            episodic_rewards = evaluators[key](ppo_agent, seed=seed)

                            episodic_rewards = np.array(episodic_rewards)
                            all_scores[key].append(episodic_rewards.mean())

                            # Get IQM for printout only
                            iqr = trimboth(episodic_rewards, proportiontocut=0.25)
                            iqr_mean = np.mean(iqr)
                            iqr_std = np.std(iqr)
                            iqr_n_samples = len(iqr)

                            eval_str += f"\n     {key} = {iqr_mean:.2f} +/- {iqr_std / np.sqrt(iqr_n_samples):.2f}"

                        print(eval_str + '\n')
                        pbar.update(1)

                continue

            # Load our dataset
            dataset_size = 1_000_000
            dataset = RecurrentReplayBufferEnv(base_env, buffer_size=dataset_size * 2)
            assert os.path.exists(f'./replay_buffers/replay_buffer_{test_id}/COMPLETE'), \
                "Replay buffer not found. Please generate dataset first."

            dataset.load(f'./replay_buffers/replay_buffer_{test_id}')

            # Print dataset stats
            dataset_sem = dataset.dataset_IQR_std / np.sqrt(dataset.dataset_IQR_n_episodes)
            mean_ep_duration = np.array(dataset.observations[0]).shape[0] / sum(dataset.dones[0])
            print(f"\n=================\nBaseline IQR return of the dataset: "
                  f"{dataset.dataset_IQR_return:.2f}"
                  f"+/- {dataset_sem:.2f}\nMean duration: {int(mean_ep_duration)} steps\n=================\n")

            # Set offline model template and training params
            offline_model = RecurrentIQL if is_iql else (RecurrentCQLSAC if is_cql else None)

            epoch_frac = 1.0
            early_stopping_limit = 10
            if 0 <= DECOY_INTERVAL <= 1:
                n_train_epochs = 50
                n_epochs_per_eval = 1
            elif DECOY_INTERVAL == 2:
                n_train_epochs = 1000
                n_epochs_per_eval = 20
                early_stopping_limit = 10
            else:
                raise ValueError("Invalid decoy interval.")

            # Log meta data
            model_save_path = f"../logs_glucose/iql_models/test_id_{test_id}/{algo}"

            print(f'\n=== Training offline for test_id {test_id} ===\n')
            for seed in experiment_seeds:

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
                    early_stopping_limit=early_stopping_limit,
                    dataset_kwargs={'epoch_fraction': epoch_frac},
                )
                # Add to our list of run scores
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
        os.makedirs('../logs_glucose/iql_logs', exist_ok=True)
        if is_iql:
            csv_path = (f'../logs_glucose/iql_logs/decoy={DECOY_INTERVAL}_expectile={EXPECTILE}'
                        f'_beta={args.beta}.csv')
        elif is_cql:
            csv_path = f'../logs_glucose/iql_logs/decoy={DECOY_INTERVAL}_CQL.csv'
        elif is_ppo:
            csv_path = f'../logs_glucose/iql_logs/ppo_evaluation.csv'
        else:
            raise ValueError("Invalid offline model type.")
        pl.DataFrame(logs).write_csv(csv_path)
