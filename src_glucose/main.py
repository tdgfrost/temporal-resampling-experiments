from collections import defaultdict
from functools import partial

import polars as pl
from scipy.stats import trim_mean
from tqdm import tqdm

from gym_wrappers import *
from models import *
from utils import *
from ppo_trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_ppo', default=False, type=parse_bool, help='Train PPO agent')
parser.add_argument('--train_offline', default=False, type=parse_bool, help='Train offline agent')

parser.add_argument('--offline_model', default='iql', type=str, choices=['iql', 'cql', 'ppo', 'random'],
                    help='Type of offline RL model to train (iql or cql)')
parser.add_argument('--target_ppo_agent', default="best_model05_33405.80.pth", type=str,
                    help='Path to target PPO agent for dataset generation')

parser.add_argument('--expectile', default=0.9, type=float, help='Expectile value for IQL training (0.5 is BC)')
parser.add_argument('--beta', default=10., type=float, help='Beta parameter for IQL agent')
parser.add_argument('--decoy_interval', default=0, type=int, help='Decoy interval: 0 (natural), 1 (1-step), 2 (2-step)')

GAMMA = 0.99
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train ids for PPO training and dataset generation
TRAIN_IDS = [i for i in range(1, 19)]
VAL_IDS = [i for i in range(19, 25)]
TEST_IDS = [i for i in range(25, 31)]

PPO_ARGS = {'train_env_creator_fn': make_glucose_env,
            'eval_env_creator_fn': make_glucose_env,
            'train_envs_per_id': 1,
            'eval_envs_per_id': 1,
            'gamma': GAMMA,
            'train_ids': TRAIN_IDS,
            'test_ids': VAL_IDS,
            'n_steps': 1024,
            'entropy_coef': 0.01,
            'clip_range': 0.2,
            'gae_lambda': 0.95,
            'n_epochs': 5,
            'hidden_dim': 128,
            'n_minibatches': 8,
            'batch_sequence_length': 8,
            'seed': MASTER_SEED,
            'learning_rate': 3e-4,
            'eval_freq': (1024 * len(TRAIN_IDS)) * 10,
            'eval_episodes': 500,
            'device': DEVICE}

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    args = parser.parse_args()

    # Establish training mode/parameters
    train_ppo = args.train_ppo
    train_offline = args.train_offline

    is_iql = args.offline_model == 'iql'
    is_cql = args.offline_model == 'cql'
    is_ppo = args.offline_model == 'ppo'  # Used for evaluation only
    is_random = args.offline_model == 'random'  # Used for evaluation only

    assert sum([train_ppo, train_offline]) == 1, \
        "Please select only one option (train_ppo, train_offline)."

    assert not train_offline or is_iql or is_cql or is_ppo or is_random, (
        "Please choose a valid offline model: 'iql', 'cql', 'ppo' (eval only), or 'random' (eval only).")

    if train_ppo:
        print(f'\n\n====== Training PPO ======\n\n')
        train_env_creator_fn = partial(make_glucose_env, use_scaling=True, enforce_ppo_wrapper=True)
        PPO_ARGS.update({'train_env_creator_fn': train_env_creator_fn})

        # *** KEY CHANGE: UPDATED HYPERPARAMETERS ***
        agent = RecurrentPPO(**PPO_ARGS)
        agent.fit(total_timesteps=3_000_000)

        # Close the envs
        agent.train_env.close()
        agent.eval_env.close()

    if train_offline:
        # --- Load our pre-trained PPO agent ----
        if is_ppo or (not is_random and not os.path.exists(f'./replay_buffer/COMPLETE')):
            ppo_agent = args.target_ppo_agent
            if ppo_agent is None:
                ppo_agent = choose_ppo_agent()
            ppo_agent = f'../logs_glucose/ppo_logs/{ppo_agent}'
            assert os.path.exists(ppo_agent), "Specified PPO agent path does not exist."
            ppo_agent = RecurrentPPO.load_checkpoint(ppo_agent, **PPO_ARGS)
        else:
            ppo_agent = None

        # --- Set up our offline training ---
        EXPECTILE = args.expectile
        DECOY_INTERVAL = args.decoy_interval
        algo_name = 'bc' if is_iql and EXPECTILE == 0.5 else args.offline_model

        if is_iql:
            print(f"\n=====\nDECOY_INTERVAL: {DECOY_INTERVAL}, EXPECTILE: {EXPECTILE}, BETA: {args.beta}\n=====\n")
        elif is_ppo:
            print('\n=====\nSpecial case - evaluating pre-trained PPO agent only.\n=====\n')
        elif is_random:
            print('\n=====\nSpecial case - evaluating random performance only.\n=====\n')
        elif is_cql:
            print(
                f"\n=====\nDECOY_INTERVAL: {DECOY_INTERVAL}\n=====\n")
        else:
            raise ValueError("Invalid offline model type.")

        logs = defaultdict(list)
        all_scores = defaultdict(list)

        logs['algo'].append(algo_name)
        logs['decoy_interval'].append(DECOY_INTERVAL)

        # Create our random seeds
        rng = np.random.default_rng(MASTER_SEED)
        n_runs = 50
        experiment_seeds = rng.integers(low=0, high=2**32 - 1, size=n_runs)

        # Establish our val and test id setup
        dummy_env = make_glucose_env()

        # Get our evaluators
        evaluators_val = dict()
        evaluators_test = dict()

        # - Online evaluation
        for key, interval in [
            ["online_irregular", 0],
            ["online_interpolated", 1],
        ]:
            evaluators_test[key] = ParallelEnvironmentEvaluator(partial(make_glucose_env,
                                                                        forced_interval=interval),
                                                                n_eval_envs=24,
                                                                n_eval_episodes_per_id=30,
                                                                gamma=GAMMA,
                                                                verbose=is_ppo or is_random,
                                                                test_ids=TEST_IDS)

            if key == "online_irregular":
                evaluators_val[key] = ParallelEnvironmentEvaluator(partial(make_glucose_env,
                                                                           forced_interval=interval),
                                                                   n_eval_envs=24,
                                                                   n_eval_episodes_per_id=20,
                                                                   gamma=GAMMA,
                                                                   verbose=False,
                                                                   test_ids=VAL_IDS)

        if DECOY_INTERVAL == 2:
            evaluators_test["online_irregular_aggregated"] = ParallelEnvironmentEvaluator(partial(make_glucose_env,
                                                                                                  forced_interval=0),
                                                                                          n_eval_envs=24,
                                                                                          n_eval_episodes_per_id=30,
                                                                                          gamma=GAMMA,
                                                                                          verbose=is_ppo or is_random,
                                                                                          test_ids=TEST_IDS)

        if is_ppo or is_random:
            print(f'\n=== Evaluating pre-trained PPO ===\n')
            with tqdm(total=n_runs) as pbar:
                for seed_idx, seed in enumerate(experiment_seeds):
                    eval_str = f"Seed {seed_idx + 1} PPO Evaluation Results:"
                    for key in evaluators_test.keys():
                        episodic_rewards = evaluators_test[key](ppo_agent, seed=seed)

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

        else:
            # Load our dataset
            dataset_size = 10_000_000
            dataset = RecurrentReplayBufferEnv(make_glucose_env(patient_ids=TRAIN_IDS), buffer_size=dataset_size * 2)

            if not os.path.exists(f'./replay_buffer/COMPLETE'):
                # Load PPO agent and fill buffer
                dataset.fill_buffer(model=ppo_agent, n_frames=dataset_size)
                dataset.save(f'./replay_buffer')
            else:
                dataset.load(f'./replay_buffer')

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
            model_save_path = f"../logs_glucose/iql_models/{algo_name}"

            for seed_idx, seed in enumerate(experiment_seeds):

                # Initialise offline model
                algo = offline_model(observation_shape=dummy_env.observation_space.shape,
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
                    evaluators_val=evaluators_val,
                    evaluators_test=evaluators_test,
                    decoy_interval=DECOY_INTERVAL,
                    early_stopping_limit=early_stopping_limit,
                    dataset_kwargs={'epoch_fraction': epoch_frac},
                )
                # Add to our list of run scores
                for key in evaluators_test.keys():
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
            csv_path = f'../logs_glucose/iql_logs/decoy={DECOY_INTERVAL}_expectile={EXPECTILE}_beta={args.beta}.csv'
        elif is_cql:
            csv_path = f'../logs_glucose/iql_logs/decoy={DECOY_INTERVAL}_CQL.csv'
        elif is_ppo:
            csv_path = f'../logs_glucose/iql_logs/ppo_baseline.csv'
        elif is_random:
            csv_path = f'../logs_glucose/iql_logs/random_baseline.csv'
        else:
            raise ValueError("Invalid offline model type.")
        pl.DataFrame(logs).write_csv(csv_path)

        # Close the evaluator envs
        for key in evaluators_test.keys():
            evaluators_test[key].eval_env.close()
        for key in evaluators_val.keys():
            evaluators_val[key].eval_env.close()
