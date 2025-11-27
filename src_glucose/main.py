from collections import defaultdict
from functools import partial

import polars as pl
from scipy.stats import trim_mean
from tqdm import tqdm
from copy import deepcopy

from gym_wrappers import *
from models import *
from utils import *
from ppo_trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_ppo', default=False, type=parse_bool, help='Train PPO agent')
parser.add_argument('--train_offline', default=False, type=parse_bool, help='Train offline agent')
parser.add_argument('--train_fqe', default=False, type=parse_bool, help='Train FQE model')

parser.add_argument('--model_type', default='dataset', type=str,
                    choices=['iql', 'cql', 'ppo', 'random', 'dataset'],
                    help='Type of model to train/evaluate (iql, cql, ppo, random)')
parser.add_argument('--target_agent_path', default=None, type=str,
                    help='Path to target agent')

parser.add_argument('--expectile', default=0.9, type=float, help='Expectile value for IQL training (0.5 is BC)')
parser.add_argument('--beta', default=10., type=float, help='Beta parameter for IQL agent')
parser.add_argument('--decoy_interval', default=0, type=int, help='Decoy interval: 0 (natural), 1 (1-step), 2 (2-step)')

GAMMA = 0.99
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

OFFLINE_ARGS = {'hidden_dim': 128,
                'recurrent_hidden_size': 128,
                'batch_size': 1024,
                'sequence_length': 64,
                'gamma': GAMMA,
                'value_lr': 3e-4,
                'policy_lr': 3e-4,
                'critic_lr': 3e-4,
                'expectile': 0.5,
                'beta': 10.0,
                'seed': 123,
                'device': DEVICE}

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    args = parser.parse_args()

    # Establish training mode/parameters
    train_ppo = args.train_ppo
    train_offline = args.train_offline
    train_fqe = args.train_fqe

    is_iql = args.model_type == 'iql'
    is_cql = args.model_type == 'cql'
    is_ppo = args.model_type == 'ppo'  # Used for evaluation only
    is_random = args.model_type == 'random'  # Used for evaluation only
    is_dataset = args.model_type == 'dataset'  # Used for evaluation only
    assert (is_dataset and train_fqe) or not is_dataset, "Dataset model type can only be used for FQE evaluation."
    model_type = 'bc' if is_iql and args.expectile == 0.5 else args.model_type

    EXPECTILE = args.expectile
    BETA = args.beta
    DECOY_INTERVAL = args.decoy_interval
    target_agent_path = args.target_agent_path

    logs = defaultdict(list)
    all_scores = defaultdict(list)
    logs['algo'].append(model_type)
    logs['decoy_interval'].append(DECOY_INTERVAL)

    # Set random seeds for experiments
    rng = np.random.default_rng(MASTER_SEED)
    n_runs = 50
    experiment_seeds = rng.integers(low=0, high=2 ** 32 - 1, size=n_runs)

    dummy_env = make_glucose_env()
    offline_model = RecurrentIQL if is_iql else (RecurrentCQLSAC if is_cql else None)
    OFFLINE_ARGS.update({'observation_shape': dummy_env.observation_space.shape,
                         'expectile': EXPECTILE,
                         'beta': BETA})

    assert sum([train_ppo, train_offline, train_fqe]) == 1, \
        "Please select only one option (train_ppo, train_offline, train_fqe)."

    assert not train_offline or is_iql or is_cql or is_ppo or is_random, (
        "Please choose a valid offline model: 'iql', 'cql', 'ppo' (eval only), or 'random' (eval only).")

    # Load the pre-trained agent if required
    ppo_agent, offline_agent, seed = None, None, MASTER_SEED
    dataset_needs_generating = False
    if train_offline or train_fqe:
        # Load PPO or offline agent
        dataset_needs_generating = train_offline and not is_random and not all([os.path.exists(f'./replay_buffer_{key}/COMPLETE')
                                                                                for key in ['train', 'val', 'test']])
        if is_ppo or dataset_needs_generating:
            if target_agent_path is None:
                target_agent_path = choose_ppo_agent()
            ppo_agent = f'../logs_glucose/ppo_logs/{target_agent_path}'
            assert os.path.exists(ppo_agent), "Specified PPO agent path does not exist."
            ppo_agent = RecurrentPPO.load_checkpoint(ppo_agent, **PPO_ARGS)

    # ===== FQE Evaluation ===== #
    if train_fqe:
        """
        We will iterate through all 50 models (for a given experiment) and train FQE ONCE (using the same random seed)
        on the relevant train_dataset. We use the validation_dataset for early stopping, and the test dataset for evaluation.
        """
        FQE_ARGS = deepcopy(OFFLINE_ARGS)
        # Update the hidden dims
        FQE_ARGS.update({'hidden_dim': 64,
                         'recurrent_hidden_size': 64,
                         'critic_lr': 5e-3})  # Higher LR for FQE training
        # Load our dataset
        datasets = load_buffer_datasets()

        # Set up our FQE evaluator
        early_stopping_key = 'fqe_evaluation_negative_loss'
        val_evaluator = {'fqe_evaluation_negative_loss': FQEEvaluator(dataset=datasets['val'],
                                                                      batch_size=1024,
                                                                      return_loss=True)}
        test_evaluator = {'fqe_evaluation': FQEEvaluator(dataset=datasets['test'],
                                                         batch_size=1024)}

        # Get our list of trained models
        if is_ppo or is_random or is_dataset:
            # Get seeds
            model_file_list = experiment_seeds
        else:
            target_model_path = f'../logs_glucose/iql_models/{model_type}/decoy_interval_{DECOY_INTERVAL}'
            model_file_list = os.listdir(target_model_path)

        for target_model_name in model_file_list:
            if is_ppo or is_random or is_dataset:
                seed = target_model_name
                if is_random or is_dataset:
                    offline_agent = CallableRandomAgentForFQE(use_dataset=is_dataset)
                else:
                    offline_agent = CallablePPOAgentForFQE(ppo_agent)
            else:
                # Extract the seed and load the pretrained agent
                seed = int(target_model_name.split('seed=')[-1].replace('.pt', ''))

                target_agent_path = os.path.join(target_model_path, target_model_name)
                OFFLINE_ARGS.update({'seed': seed})
                offline_agent = offline_model(**OFFLINE_ARGS)
                offline_agent.load_checkpoint(target_agent_path)

            # Create our FQE model
            FQE_ARGS.update({'target_model': offline_agent})
            algo = RecurrentFQE(**FQE_ARGS)

            # Fit our FQE model
            log_dict = algo.fit(
                dataset=datasets['train'],
                accessory_datasets=[datasets[key] for key in ['val', 'test']],
                n_epochs_train=50,
                n_epochs_per_eval=1,
                evaluators_val=val_evaluator,
                evaluators_test=test_evaluator,
                early_stopping_key=early_stopping_key,
                decoy_interval=DECOY_INTERVAL,
                early_stopping_limit=10,
                dataset_kwargs={},
            )

            # Add to our list of run scores
            for key in test_evaluator.keys():
                all_scores[key].append(log_dict[key])

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
        os.makedirs('../logs_glucose/fqe_logs', exist_ok=True)
        csv_path = f'../logs_glucose/fqe_logs/decoy={DECOY_INTERVAL}_{model_type}.csv'
        pl.DataFrame(logs).write_csv(csv_path)

    # ===== Training PPO ===== #
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

    # ===== Training Offline Agent (BC/IQL/CQL) ===== #
    if train_offline:
        # --- Set up our offline training ---
        if is_iql:
            print(f"\n=====\nDECOY_INTERVAL: {DECOY_INTERVAL}, EXPECTILE: {EXPECTILE}, BETA: {BETA}\n=====\n")
        elif is_ppo:
            print('\n=====\nSpecial case - evaluating pre-trained PPO agent only.\n=====\n')
        elif is_random:
            print('\n=====\nSpecial case - evaluating random performance only.\n=====\n')
        elif is_cql:
            print(
                f"\n=====\nDECOY_INTERVAL: {DECOY_INTERVAL}\n=====\n")
        else:
            raise ValueError("Invalid offline model type.")

        # Get our evaluators
        evaluators_val = dict()
        evaluators_test = dict()

        # - Online evaluation
        early_stopping_key = 'online_irregular_IQM'
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
                                                                   n_eval_envs=30,
                                                                   n_eval_episodes_per_id=5,
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
            datasets = load_buffer_datasets(fill_if_absent=dataset_needs_generating, ppo_agent=ppo_agent)
            train_dataset = datasets['train']

            # Print dataset stats
            dataset_sem = train_dataset.dataset_IQR_std / np.sqrt(train_dataset.dataset_IQR_n_episodes)
            mean_ep_duration = np.array(train_dataset.observations[0]).shape[0] / sum(train_dataset.dones[0])
            print(f"\n=================\nBaseline IQR return of the dataset: "
                  f"{train_dataset.dataset_IQR_return:.2f}"
                  f"+/- {dataset_sem:.2f}\nMean duration: {int(mean_ep_duration)} steps\n=================\n")

            # Set offline model template and training params
            early_stopping_limit = 10 if DECOY_INTERVAL == 2 and not is_cql else 5
            n_train_epochs = 50
            n_epochs_per_eval = 1

            # Log meta data
            model_save_path = f"../logs_glucose/iql_models/{model_type}/decoy_interval_{DECOY_INTERVAL}"

            for seed_idx, seed in enumerate(experiment_seeds):

                print(f'\n========== Starting seed {seed_idx + 1}/{len(experiment_seeds)} ==========\n')

                # Initialise offline model
                OFFLINE_ARGS.update({'seed': seed})
                algo = offline_model(**OFFLINE_ARGS)

                log_dict = algo.fit(
                    dataset=train_dataset,
                    n_epochs_train=n_train_epochs,
                    n_epochs_per_eval=n_epochs_per_eval,
                    evaluators_val=evaluators_val,
                    evaluators_test=evaluators_test,
                    early_stopping_key=early_stopping_key,
                    decoy_interval=DECOY_INTERVAL,
                    early_stopping_limit=early_stopping_limit,
                    dataset_kwargs={},
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
            csv_path = f'../logs_glucose/iql_logs/decoy={DECOY_INTERVAL}_expectile={EXPECTILE}_beta={BETA}.csv'
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
