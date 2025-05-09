\
import boto3
import numpy as np
from sklearn.model_selection import train_test_split # Changed from TimeSeriesSplit
import json
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math
import traceback
import optuna
from functools import reduce
import operator
import time

class GridSearchOpt1Partition:
    """
    Class to perform grid search optimization for a single data partition using Optuna's GridSampler.
    Accepts parameter configuration defining ranges/values and generates the grid.
    Each hyperparameter set is evaluated on a single train-test split.
    Results are uploaded periodically to the cloud if enabled.
    """
    def __init__(self, X, y, model, metric,
                 param_config: dict,
                 save_path: str,
                 cloud_name: str, cloud_bucket: str, cloud_key: str,
                 test_size=0.2, # For train_test_split
                 minimize=True,
                 upload_cloud_rate=100,
                 n_jobs=1, # Optuna n_jobs for parallel trials (each trial is one param set)
                 upload_to_cloud=True,
                 random_state_split=None): # Added for reproducibility of split
        self.X = X
        self.y = y
        self.model_cls = model
        self.metric = metric
        self.param_config = param_config
        self.save_path = save_path
        self.cloud_name = cloud_name
        self.cloud_bucket = cloud_bucket
        self.cloud_key = cloud_key
        self.test_size = test_size
        self.random_state_split = random_state_split
        self.minimize = minimize
        self.upload_cloud_rate = upload_cloud_rate
        self.upload_executor = ThreadPoolExecutor(max_workers=4)
        self.upload_futures = []
        self.n_jobs = n_jobs
        self.upload_to_cloud = upload_to_cloud
        self.stationary_params = {}
        self._ensure_save_dir()

        # Perform the single split here to be used by all trials
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            shuffle=False, # Typically False for time-series like data, or can be configurable
            random_state=self.random_state_split
        )

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
        if not self.upload_to_cloud:
            print("Cloud upload is disabled. Skipping upload.")
            return

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
        if not self.upload_to_cloud:
            print("Cloud upload is disabled. Skipping waiting for uploads.")
            return

        for future in as_completed(self.upload_futures):
            try:
                future.result()
            except Exception as e:
                print(f"An upload task failed: {e}")
        self.upload_futures = []

    def _generate_optuna_grid_search_space(self):
        search_space = {}
        self.stationary_params = {}
        print("Generating Optuna GridSampler search space and identifying stationary params:")
        for name, config_tuple in self.param_config.items():
            if len(config_tuple) != 4:
                raise ValueError(f"Parameter config for '{name}' must be a 4-element tuple: (low/values, high, num/None, type)")

            item1, item2, item3, ptype = config_tuple
            ptype = ptype.lower()

            if ptype == 'float':
                low, high, num_points = item1, item2, item3
                if not isinstance(num_points, int) or num_points < 1:
                    raise ValueError(f"Number of points ('num') for float param '{name}' must be a positive integer.")
                values = np.linspace(low, high, num_points).tolist()
                search_space[name] = values
                print(f"  - {name} (float): {len(values)} points from {low} to {high}")
            elif ptype == 'int':
                low, high, num_points = item1, item2, item3
                if not isinstance(num_points, int) or num_points < 1:
                    raise ValueError(f"Number of points ('num') for int param '{name}' must be a positive integer.")
                if high < low:
                    raise ValueError(f"High ({high}) must be >= low ({low}) for int param '{name}'")
                if high - low + 1 >= num_points: # if range allows for num_points steps
                    vals = np.linspace(low, high, num_points)
                    values = sorted(list(np.unique(np.round(vals).astype(int))))
                else: # if num_points is larger than the actual number of integers in range
                    values = list(np.arange(low, high + 1))
                search_space[name] = values
                print(f"  - {name} (int): {len(values)} points from {low} to {high} -> {values}")
            elif ptype == 'cat':
                cat_values = item1
                if not isinstance(cat_values, (list, tuple)):
                    raise ValueError(f"Categorical values ('low/values') for param '{name}' must be a list or tuple.")
                values = list(cat_values)
                search_space[name] = values
                print(f"  - {name} (cat): {len(values)} values -> {values}")
            elif ptype == 'stationary':
                value = item1 # For stationary, 'low/values' holds the fixed value
                self.stationary_params[name] = value
                print(f"  - {name} (stationary): Fixed value -> {value}")
            else:
                raise ValueError(f"Unsupported parameter type: '{ptype}' for param '{name}'")
        return search_space

    def save_result(self, trial_number, score, params, computation_time, non_optimal_count):
        save_path = self.save_path
        if isinstance(score, float):
            if math.isnan(score):
                score = "NaN"
            elif math.isinf(score):
                score = "Infinity" if score > 0 else "-Infinity"
        elif not isinstance(score, (int, str)):
            score = str(score)

        serializable_params = {
            k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))
        }

        result_data = {
            "trial": trial_number,
            "target": score,
            "params": serializable_params,
            "computation_time_seconds": computation_time,
            "non_optimal_fits": non_optimal_count
        }
        try:
            with open(save_path, 'a') as f:
                json.dump(result_data, f)
                f.write('\n')
        except IOError as e:
            print(f"Error saving trial result to {save_path}: {e}")
        except TypeError as e:
            print(f"Error serializing trial result {trial_number}: {result_data}. Error: {e}")

    def optimize_single_split(self):
        try:
            search_space = self._generate_optuna_grid_search_space()
            if search_space:
                 total_trials = reduce(operator.mul, [len(v) for v in search_space.values()], 1)
            else: # Only stationary params
                 total_trials = 1
            if total_trials == 0 and search_space:
                print("Generated search space for varying parameters is empty (total_trials=0). Exiting.")
                return
        except Exception as e:
            print(f"Error generating search space or calculating total trials: {e}")
            traceback.print_exc()
            return

        sampler = optuna.samplers.GridSampler(search_space) if search_space else None
        direction = "minimize" if self.minimize else "maximize"

        def objective(trial):
            suggested_params = {}
            for name, values in search_space.items(): # Only suggest for varying params
                suggested_params[name] = trial.suggest_categorical(name, values)

            params = {**self.stationary_params, **suggested_params}

            score = None
            non_optimal_count = 0
            start_time = time.time()

            try:
                model_instance = self.model_cls(**params)
                model_instance.fit(self.X_train, self.y_train)

                if hasattr(model_instance, 'status') and model_instance.status != 'optimal':
                    non_optimal_count += 1
                    print(f"    Warning: Non-optimal status ('{model_instance.status}') for params {params}.")

                y_pred = model_instance.predict(self.X_val).flatten()
                current_score = self.metric(self.y_val, y_pred)

                if not isinstance(current_score, (int, float)) or math.isnan(current_score) or math.isinf(current_score):
                    print(f"    Warning: Invalid score ({current_score}) for params {params}. Treating as failure.")
                    score = float('inf') if self.minimize else float('-inf') # Penalize
                else:
                    score = float(current_score)

            except Exception as e:
                print(f"    Error evaluating params {params}: {e}")
                score = float('inf') if self.minimize else float('-inf') # Penalize on error

            end_time = time.time()
            computation_time = end_time - start_time
            
            # Save result for the trial
            # Combine stationary and suggested params for saving
            full_params_for_saving = {**self.stationary_params, **suggested_params}
            self.save_result(trial.number, score if score is not None else "EvaluationFailed", full_params_for_saving, computation_time, non_optimal_count)

            if (trial.number + 1) % self.upload_cloud_rate == 0:
                print(f"\nTrial {trial.number + 1}: Reached upload threshold ({self.upload_cloud_rate}). Triggering upload...")
                self._trigger_upload()
            
            if score is None: # Should not happen if penalized above, but as a safeguard
                 raise optuna.exceptions.TrialPruned("Evaluation failed, score is None")
            return score

        print("--- Starting Optuna Study (Single Split Grid Search) ---")
        print(f"  Stationary parameters: {self.stationary_params}")
        if search_space:
            print(f"  Evaluating {total_trials} combinations for varying parameters on a single split (n_jobs={self.n_jobs})")
        else:
            print(f"  Evaluating 1 combination (only stationary params) on a single split (n_jobs={self.n_jobs})")


        study = optuna.create_study(sampler=sampler, direction=direction) if sampler else optuna.create_study(direction=direction)

        try:
            study.optimize(objective, n_trials=total_trials, n_jobs=self.n_jobs, catch=(Exception,), show_progress_bar=True)
        except Exception as e:
            print(f"!! Optuna study optimize failed: {e}")
            traceback.print_exc()

        print("--- Finished Optuna Study ---")

        try:
            best_trial_info = study.best_trial
            print(f"  Best trial: Score={best_trial_info.value}, Params={best_trial_info.params}")
        except ValueError: # No trials completed
            print("  No completed trials found in the study.")

        print("Study complete. Triggering final upload...")
        self._trigger_upload()
        print("Waiting for all uploads to finish...")
        self._wait_for_uploads()
        print("All uploads finished.")


if __name__ == "__main__":
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    # Generate sample data
    X_data = np.random.randn(100, 5)
    y_data = X_data[:, 0] * 2.5 + X_data[:, 1] * -1.5 + np.random.randn(100) * 1.2

    def example_metric_mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    # Define parameter configuration for grid search
    # Format: (low/values, high, num_points/None, type)
    # 'type' can be 'float', 'int', 'cat', 'stationary'
    param_config_example = {
        'alpha': (0.01, 10.0, 3, 'float'),      # 3 float values from 0.01 to 10.0
        'solver': (['svd', 'cholesky', 'lsqr'], None, None, 'cat'), # Categorical
        'fit_intercept': (True, None, None, 'stationary'), # Stationary boolean
        'tol': (1e-4, 1e-2, 2, 'float') # 2 float values for tolerance
    }
    # Expected combinations: 3 (alpha) * 3 (solver) * 2 (tol) = 18 trials

    results_file_path = "grid_search_1_partition_results.jsonl"
    cloud_provider_name = "gcp"  # or "aws", or other to skip
    cloud_storage_bucket = "your-gcp-bucket-name" # Replace with your bucket name
    cloud_storage_key = "optimization_tests/grid_search_1_partition.jsonl" # Replace

    # Clean up previous results file if it exists
    if os.path.exists(results_file_path):
        os.remove(results_file_path)

    upload_frequency = 5 # Upload results to cloud every 5 trials

    # Initialize the optimizer
    optimizer_instance = GridSearchOpt1Partition(
        X=X_data,
        y=y_data,
        model=Ridge, # Scikit-learn model class
        metric=example_metric_mse,
        param_config=param_config_example,
        save_path=results_file_path,
        cloud_name=cloud_provider_name,
        cloud_bucket=cloud_storage_bucket,
        cloud_key=cloud_storage_key,
        test_size=0.25, # 25% of data for validation
        minimize=True, # True if the metric should be minimized (e.g., MSE)
        upload_cloud_rate=upload_frequency,
        n_jobs=1, # Use 1 job for simplicity in example, can be >1 for parallel trials
        upload_to_cloud=False, # Set to True to enable actual cloud uploads
        random_state_split=42 # For reproducible train/test split
    )

    # Run the optimization
    optimizer_instance.optimize_single_split()

    print("\n--- Grid Search (1 Partition) Optimization Complete ---")
    print(f"All trial results saved to {results_file_path}")
    if optimizer_instance.upload_to_cloud:
        print(f"Results uploaded periodically to {cloud_provider_name} bucket {cloud_storage_bucket} at {cloud_storage_key}")
    else:
        print("Cloud upload was disabled for this run.")

    # You can now inspect 'grid_search_1_partition_results.jsonl' for the results.
    # If upload_to_cloud was True and credentials configured, results would also be in the cloud.
