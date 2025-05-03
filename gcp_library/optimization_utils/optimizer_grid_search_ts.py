# %%
import boto3
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import json
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math
import traceback
import optuna
from functools import reduce
import operator
import time # Add time import

class TimeSeriesOpt:
    """
    Class to perform grid search optimization using Optuna's GridSampler.
    Accepts parameter configuration defining ranges/values and generates the grid.
    Each hyperparameter set is evaluated across ALL time series splits,
    and the MEAN score is saved. Results are uploaded periodically.
    """
    def __init__(self, X, y, model, metric,
                 param_config: dict, # Changed from param_grid
                 save_path: str,
                 cloud_name: str, cloud_bucket: str, cloud_key: str,
                 n_splits=5, test_size=None, gap=0,
                 minimize=True,
                 upload_cloud_rate=100,
                 n_jobs=-1):
        self.X = X
        self.y = y
        self.model_cls = model
        self.metric = metric
        self.param_config = param_config # Store the new config format
        self.save_path = save_path
        self.cloud_name = cloud_name
        self.cloud_bucket = cloud_bucket
        self.cloud_key = cloud_key
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        self.minimize = minimize
        self.upload_cloud_rate = upload_cloud_rate
        self.upload_executor = ThreadPoolExecutor(max_workers=4)
        self.upload_futures = []
        self.n_jobs = n_jobs
        self.tscv_indices = list(self.tscv.split(self.X))
        self._ensure_save_dir()
        self.stationary_params = {} # Initialize dictionary for stationary params

    def _ensure_save_dir(self):
        save_dir = os.path.dirname(self.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def upload_to_s3(self):
        print(f"Attempting to upload {self.save_path} to S3 bucket {self.cloud_bucket}...")
        try:
            s3 = boto3.client('s3')
            s3.upload_file(self.save_path, self.cloud_bucket, self.cloud_key)
            print("S3 upload successful.")
        except Exception as e:
            print(f"S3 upload failed: {e}")

    def upload_to_gstorage(self):
        print(f"Attempting to upload {self.save_path} to GCS bucket {self.cloud_bucket}...")
        try:
            client = storage.Client()
            bucket = client.bucket(self.cloud_bucket)
            blob = bucket.blob(self.cloud_key)
            blob.upload_from_filename(self.save_path)
            print("GCS upload successful.")
        except Exception as e:
            print(f"GCS upload failed: {e}")

    def _trigger_upload(self):
        if self.cloud_name.lower() in ["amazon", "aws"]:
            future = self.upload_executor.submit(self.upload_to_s3)
            self.upload_futures.append(future)
            print("File upload to S3 submitted.", flush=True)
        elif self.cloud_name.lower() in ["google", "gcp"]:
            future = self.upload_executor.submit(self.upload_to_gstorage)
            self.upload_futures.append(future)
            print("File upload to GCS submitted.", flush=True)
        else:
            print(f"Warning: Unsupported cloud provider: {self.cloud_name}. Skipping upload.")

    def _wait_for_uploads(self):
        for future in as_completed(self.upload_futures):
            try:
                future.result()
            except Exception as e:
                print(f"An upload task failed: {e}")
        self.upload_futures = []

    def _generate_optuna_grid_search_space(self):
        """
        Generates the search_space dict for Optuna GridSampler from param_config
        and separates stationary parameters.
        """
        search_space = {}
        self.stationary_params = {} # Reset stationary params
        print("Generating Optuna GridSampler search space and identifying stationary params:")
        for name, config_tuple in self.param_config.items():
            if len(config_tuple) != 4:
                raise ValueError(f"Parameter config for '{name}' must be a 4-element tuple: (low, high, num/values, type)")

            low, high, num_or_values, ptype = config_tuple
            ptype = ptype.lower()

            if ptype == 'float':
                if not isinstance(num_or_values, int) or num_or_values < 1:
                    raise ValueError(f"Number of points ('num') for float param '{name}' must be a positive integer.")
                values = np.linspace(low, high, num_or_values).tolist()
                search_space[name] = values
                print(f"  - {name} (float): {len(values)} points from {low} to {high}")
            elif ptype == 'int':
                if not isinstance(num_or_values, int) or num_or_values < 1:
                    raise ValueError(f"Number of points ('num') for int param '{name}' must be a positive integer.")
                if high < low:
                    raise ValueError(f"High ({high}) must be >= low ({low}) for int param '{name}'")
                if high - low + 1 >= num_or_values:
                    vals = np.linspace(low, high, num_or_values)
                    values = sorted(list(np.unique(np.round(vals).astype(int))))
                else:
                    values = list(np.arange(low, high + 1))
                search_space[name] = values
                print(f"  - {name} (int): {len(values)} points from {low} to {high} -> {values}")
            elif ptype == 'cat':
                if not isinstance(low, (list, tuple)):
                    raise ValueError(f"Categorical values ('low') for param '{name}' must be a list or tuple.")
                values = list(low)
                search_space[name] = values
                print(f"  - {name} (cat): {len(values)} values -> {values}")
            elif ptype == 'stationary':
                value = low # For stationary, 'low' holds the fixed value
                self.stationary_params[name] = value
                print(f"  - {name} (stationary): Fixed value -> {value}")
            else:
                raise ValueError(f"Unsupported parameter type: '{ptype}' for param '{name}'")

        return search_space

    def save_result(self, trial_number, mean_score, params, computation_time, non_optimal_count): # Add non_optimal_count parameter
        """Saves the mean score, computation time, and non-optimal fit count of a trial to the JSONL file."""
        save_path = self.save_path
        if isinstance(mean_score, float):
            if math.isnan(mean_score): mean_score = "NaN"
            elif math.isinf(mean_score): mean_score = "Infinity" if mean_score > 0 else "-Infinity"
        elif not isinstance(mean_score, (int, str)):
            mean_score = str(mean_score)

        serializable_params = {
            k: v for k, v in params.items() if isinstance(v, (int, float, str, type(None)))
        }

        result_data = {
            "trial": trial_number,
            "target": mean_score,
            "params": serializable_params,
            "computation_time_seconds": computation_time,
            "non_optimal_fits": non_optimal_count # Add non-optimal count to saved data
        }
        try:
            with open(save_path, 'a') as f:
                json.dump(result_data, f)
                f.write('\n')
        except IOError as e:
            print(f"Error saving trial result to {save_path}: {e}")
        except TypeError as e:
            print(f"Error serializing trial result {trial_number}: {result_data}. Error: {e}")

    def optimize_splits(self):
        """
        Performs grid search using Optuna GridSampler and parallel trials.
        Each trial evaluates params across all splits, saves the mean score,
        and uploads periodically. Stationary parameters are fixed.
        """
        try:
            search_space = self._generate_optuna_grid_search_space()
            # Calculate total trials based only on the varying parameters in search_space
            if search_space:
                 total_trials = reduce(operator.mul, [len(v) for v in search_space.values()], 1)
            else:
                 total_trials = 1 # If only stationary params, run once
            if total_trials == 0 and search_space: # Check if search space is non-empty but resulted in 0 trials
                print("Generated search space for varying parameters is empty (total_trials=0). Exiting.")
                return
        except Exception as e:
            print(f"Error generating search space or calculating total trials: {e}")
            traceback.print_exc() # Print traceback for detailed error
            return

        sampler = optuna.samplers.GridSampler(search_space) if search_space else None # Use GridSampler only if there are varying params
        direction = "minimize" if self.minimize else "maximize"

        def objective(trial):
            # Suggest values only for parameters in the search_space
            suggested_params = {}
            for name, values in search_space.items():
                suggested_params[name] = trial.suggest_categorical(name, values)

            # Combine suggested params with fixed stationary params
            params = {**self.stationary_params, **suggested_params}

            split_scores = []
            non_optimal_count = 0 # Initialize non-optimal counter
            start_time = time.time() # Start timer
            for split_idx, (train_index, test_index) in enumerate(self.tscv_indices):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                try:
                    model_instance = self.model_cls(**params)
                    model_instance.fit(X_train, y_train)

                    # Check model status if the attribute exists
                    if hasattr(model_instance, 'status') and model_instance.status != 'optimal':
                        non_optimal_count += 1
                        print(f"    Warning: Non-optimal status ('{model_instance.status}') on split {split_idx} for params {params}.")

                    y_pred = model_instance.predict(X_test).flatten()
                    score = self.metric(y_test, y_pred)

                    if not isinstance(score, (int, float)) or math.isnan(score) or math.isinf(score):
                        print(f"    Warning: Invalid score ({score}) on split {split_idx} for params {params}. Skipping split.")
                        continue
                    else:
                        split_scores.append(float(score))

                except Exception as e:
                    print(f"    Error evaluating params {params} on split {split_idx}: {e}")
                    continue

            end_time = time.time() # End timer
            computation_time = end_time - start_time # Calculate duration

            if not split_scores:
                print(f"    Warning: No valid scores obtained across any split for params {params}. Pruning trial.")
                # Save result even if pruned, including non_optimal_count
                # Combine stationary and suggested params for saving
                full_params_for_saving = {**self.stationary_params, **suggested_params}
                self.save_result(trial.number, "AllSplitsFailed", full_params_for_saving, computation_time, non_optimal_count)
                raise optuna.exceptions.TrialPruned()

            mean_score = np.mean(split_scores)
            # Pass computation_time and non_optimal_count to save_result
            # Combine stationary and suggested params for saving
            full_params_for_saving = {**self.stationary_params, **suggested_params}
            self.save_result(trial.number, mean_score, full_params_for_saving, computation_time, non_optimal_count)

            if (trial.number + 1) % self.upload_cloud_rate == 0:
                print(f"\nTrial {trial.number + 1}: Reached upload threshold ({self.upload_cloud_rate}). Triggering upload...")
                self._trigger_upload()

            return mean_score

        print(f"--- Starting Optuna Study ---")
        print(f"  Stationary parameters: {self.stationary_params}")
        if search_space:
            print(f"  Evaluating {total_trials} combinations for varying parameters across {self.n_splits} splits each (n_jobs={self.n_jobs})")
        else:
            print(f"  Evaluating 1 combination (only stationary params) across {self.n_splits} splits each (n_jobs={self.n_jobs})")

        # Create study with or without sampler based on whether search_space exists
        study = optuna.create_study(sampler=sampler, direction=direction) if sampler else optuna.create_study(direction=direction)

        try:
            # Adjust n_trials based on calculated total_trials
            study.optimize(objective, n_trials=total_trials, n_jobs=self.n_jobs, catch=(Exception,), show_progress_bar=True)
        except Exception as e:
            print(f"!! Optuna study optimize failed: {e}")
            traceback.print_exc() # Print traceback

        print(f"--- Finished Optuna Study ---")

        try:
            best_trial_info = study.best_trial
            print(f"  Best trial overall: Score={best_trial_info.value}, Params={best_trial_info.params}")
        except ValueError:
            print(f"  No completed trials found in the study.")

        print("Study complete. Triggering final upload (if needed) and waiting for all uploads...")
        self._wait_for_uploads()
        print("All uploads finished.")

# %% Example usage
if __name__ == "__main__":
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    X = np.random.randn(150, 5)
    y = X[:, 0] * 3 + X[:, 2] * -2 + np.random.randn(150) * 1.5

    def example_metric(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    param_config = {
        'alpha': (0.01, 100.0, 5, 'float'),
        'solver': (['svd', 'cholesky', 'lsqr', 'sag', 'saga'], None, None, 'cat'),
        'fit_intercept': (True, None, None, 'stationary'), # Example stationary param
        'copy_X': (True, None, None, 'stationary') # Another stationary param
    }

    save_file_path = "grid_search_optuna_split_results.jsonl"
    cloud_provider = "gcp"
    bucket = "svr_hyper_opt"
    key = "optimization_results_test_1/grid_search_optuna_split.jsonl"

    if os.path.exists(save_file_path):
        os.remove(save_file_path)

    upload_rate_trials = 50

    optimizer = TimeSeriesOpt(
        X=X,
        y=y,
        model=Ridge,
        metric=example_metric,
        param_config=param_config,
        save_path=save_file_path,
        cloud_name=cloud_provider,
        cloud_bucket=bucket,
        cloud_key=key,
        n_splits=5,
        test_size=15,
        gap=0,
        minimize=True,
        upload_cloud_rate=upload_rate_trials,
        n_jobs=4
    )

    optimizer.optimize_splits()

    print("\n--- Grid Search Optimization Complete ---")
    print(f"All trial results (mean score across splits) saved to {save_file_path}")
    print(f"Results uploaded periodically to {cloud_provider} bucket {bucket} at {key}")

# %%


