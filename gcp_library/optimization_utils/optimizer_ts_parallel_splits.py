\
# %%
import boto3
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import optuna
import json
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import time # Added for potential timing within __call__ if needed

class TimeSeriesOptParallelSplits:
    """
    Class to define the function used for Bayesian Optimization
    for time series data, parallelizing evaluation across time-series splits.
    """
    def __init__(self, X, y, model_class, metric_func, save_path: str, 
                 cloud_name: str, cloud_bucket: str, cloud_key: str, 
                 n_splits=5, test_size=None, gap=0, upload_cloud_rate=100, 
                 direction="minimize", n_jobs_splits=1, upload_to_cloud=True, **hyperparams_config):
        self.X = X
        self.y = y
        self.model_class = model_class
        self.metric_func = metric_func
        self.save_path = save_path
        self.cloud_name = cloud_name
        self.cloud_bucket = cloud_bucket
        self.cloud_key = cloud_key
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.hyperparams_config = hyperparams_config # Renamed from params to avoid confusion
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size, gap=self.gap)
        self.upload_cloud_rate = upload_cloud_rate
        
        # n_jobs_splits controls parallelism for splits within a trial
        self.n_jobs_splits = n_jobs_splits 
        self.split_executor = ThreadPoolExecutor(max_workers=self.n_jobs_splits)
        self.upload_executor = ThreadPoolExecutor(max_workers=4) # For cloud uploads
        self.upload_futures = []
        
        self.direction = direction
        self.upload_to_cloud = upload_to_cloud

    def _save_trial_result(self, params_to_save, mean_score):
        with open(self.save_path, 'a') as f:
            json.dump({"target": mean_score, "params": params_to_save}, f)
            f.write('\\n')

    def _upload_to_s3(self):
        # Ensure boto3 is imported if not at the top level or handle missing import
        try:
            s3 = boto3.client('s3')
            s3.upload_file(self.save_path, self.cloud_bucket, self.cloud_key)
            print(f"Successfully uploaded {self.save_path} to S3 bucket {self.cloud_bucket} at {self.cloud_key}", flush=True)
        except Exception as e:
            print(f"S3 upload failed: {e}", flush=True)
        
    def _upload_to_gstorage(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.cloud_bucket)
            blob = bucket.blob(self.cloud_key)
            blob.upload_from_filename(self.save_path)
            print(f"Successfully uploaded {self.save_path} to GCS bucket {self.cloud_bucket} at {self.cloud_key}", flush=True)
        except Exception as e:
            print(f"GCS upload failed: {e}", flush=True)

    def _trigger_cloud_upload(self):
        if not self.upload_to_cloud:
            return

        if self.cloud_name.lower() in ["amazon", "aws"]:
            future = self.upload_executor.submit(self._upload_to_s3)
            self.upload_futures.append(future)
            print("File upload to S3 submitted.", flush=True)
        elif self.cloud_name.lower() in ["google", "gcp"]:
            future = self.upload_executor.submit(self._upload_to_gstorage)
            self.upload_futures.append(future)
            print("File upload to GCS submitted.", flush=True)
        else:
            print(f"Unsupported cloud provider: {self.cloud_name}. Skipping upload.", flush=True)

    def _wait_for_pending_uploads(self):
        if not self.upload_to_cloud:
            return
        
        for future in as_completed(self.upload_futures):
            try:
                future.result() # Wait for the upload to finish and retrieve result/exception
            except Exception as e:
                print(f"An upload task failed: {e}", flush=True)
        self.upload_futures = [] # Clear the list after processing

    def _evaluate_single_split(self, train_index, test_index, current_hyperparams):
        X_train, X_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        
        model_instance = self.model_class(**current_hyperparams)
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test).flatten()
        
        score = self.metric_func(y_test, y_pred)
        try:
            # Optuna typically maximizes. If metric is error (e.g., MAE, MSE),
            # it should be minimized. If it's accuracy, it should be maximized.
            # This check is specific to MAPE.
            if self.metric_func.__name__ == "mean_absolute_percentage_error":
                score *= 100
        except AttributeError:
            # self.metric_func might be a functools.partial or other wrapper
            pass 
            
        return score

    def __call__(self, trial: optuna.Trial):
        current_hyperparams = {}
        for param_name, config in self.hyperparams_config.items():
            low, high, ptype = config # Assuming config is (low, high, type) or (values, None, "cat")
            if ptype == "float":
                current_hyperparams[param_name] = trial.suggest_float(param_name, low, high)
            elif ptype == "int":
                current_hyperparams[param_name] = trial.suggest_int(param_name, low, high)
            elif ptype == "cat":
                current_hyperparams[param_name] = trial.suggest_categorical(param_name, low) # low is list of categories
            elif ptype == "stationary":
                current_hyperparams[param_name] = low # low is the fixed value
            else:
                raise ValueError(f"Unsupported parameter type: {ptype} for {param_name}")
        
        split_futures = []
        for train_idx, test_idx in self.tscv.split(self.X):
            future = self.split_executor.submit(self._evaluate_single_split, train_idx, test_idx, current_hyperparams)
            split_futures.append(future)
        
        scores = []
        for future in as_completed(split_futures):
            try:
                scores.append(future.result())
            except Exception as e:
                print(f"Error evaluating a split: {e}", flush=True)
                # Decide how to handle split errors: skip, use NaN, or raise
                # For now, let's append NaN if Optuna handles it, or skip.
                # Optuna prefers finite numbers. If a split fails, it might be best to return a very bad score.
                # Or, if many fail, the trial itself might be considered failed.
                # For simplicity, we'll let it propagate or be caught by Optuna if it raises.
                # If we want to make it robust, return a score indicating failure.
                # For now, if a split fails, it might cause np.mean to be problematic if not handled.
                # Let's assume for now that future.result() will raise, and Optuna might prune this trial.
                # Or, we can return a score that Optuna will consider "bad".
                # Example: if self.direction == "minimize", return float('inf')
                if self.direction == "minimize":
                    return float('inf') 
                else:
                    return -float('inf')


        if not scores: # All splits failed
             print(f"Trial {trial.number}: All splits failed evaluation.", flush=True)
             if self.direction == "minimize":
                return float('inf') 
             else:
                return -float('inf')

        mean_score = np.mean(scores)
        
        # Prepare params for saving (only serializable ones)
        params_to_save = {
            name: val for name, val in current_hyperparams.items()
            if isinstance(val, (int, float, str, bool)) or val is None
        }
        
        self._save_trial_result(params_to_save, mean_score)
        print(f"Trial {trial.number}: Mean Score = {mean_score:.4f}, Params = {params_to_save}", flush=True)
        
        if (trial.number + 1) % self.upload_cloud_rate == 0:
            self._trigger_cloud_upload()
        
        # Wait for any uploads triggered in this trial or previous ones if they are still pending
        # This ensures that if an upload was triggered, it completes before the next trial potentially modifies the file.
        # However, _wait_for_pending_uploads clears the list. We only want to wait for the one just submitted.
        # A better approach might be to wait for all at the end of optimize, or manage futures more carefully.
        # For now, let's stick to the original logic of waiting if any are pending.
        self._wait_for_pending_uploads() # Waits for ALL submitted futures in self.upload_futures
        
        return mean_score

    def optimize(self, n_trials: int):
        study = optuna.create_study(direction=self.direction)
        # Parallelism is handled within __call__ for splits, so Optuna trials run sequentially.
        study.optimize(self, n_trials=n_trials, n_jobs=1) 
        
        print("\\nOptimization Finished!", flush=True)
        print(f"Best trial ({study.best_trial.number}):", flush=True)
        print(f"  Value: {study.best_value:.4f}", flush=True)
        print("  Params: ", flush=True)
        for key, value in study.best_params.items():
            print(f"    {key}: {value}", flush=True)
        
        # Final upload if pending and enabled
        if self.upload_to_cloud and self.upload_futures:
            print("Waiting for final uploads to complete...", flush=True)
            self._wait_for_pending_uploads()
            print("All uploads finished.", flush=True)
            
        return study

# %% Example Usage
if __name__ == "__main__":
    # Ensure necessary imports for the example
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import os

    # Create dummy data
    X_data = np.random.rand(100, 5)
    y_data = np.random.rand(100) * 10

    # Define a simple model class for the example
    class ExampleModel:
        def __init__(self, n_estimators=100, max_depth=None, random_state=None):
            self.model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=random_state
            )
        
        def fit(self, X_train, y_train):
            self.model.fit(X_train, y_train)
            
        def predict(self, X_test):
            return self.model.predict(X_test)

    # Define a metric function
    def example_metric(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) # Lower is better

    # Define hyperparameter configuration
    # Format: param_name: (low, high_or_values, type)
    hyperparams_config_example = {
        'n_estimators': (50, 200, "int"),
        'max_depth': (3, 10, "int"),
        # 'criterion': (['mse', 'mae'], None, "cat") # For sklearn > 0.24, now 'squared_error', 'absolute_error'
        'criterion': (['squared_error', 'absolute_error'], None, "cat"),
        'random_state': (42, None, "stationary") # Fixed value
    }
    
    results_file = "ts_parallel_splits_results.jsonl"
    # Clean up previous results file if it exists for a fresh run
    if os.path.exists(results_file):
        os.remove(results_file)

    optimizer = TimeSeriesOptParallelSplits(
        X=X_data,
        y=y_data,
        model_class=ExampleModel,
        metric_func=example_metric,
        save_path=results_file,
        cloud_name="gcp",  # or "aws", or other to skip actual upload
        cloud_bucket="your-gcp-bucket-name", # Replace with your bucket
        cloud_key="optimization_output/ts_parallel_splits.jsonl", # Replace
        n_splits=5,
        test_size=20, # 20 samples per test split
        gap=5,
        upload_cloud_rate=5, # Upload every 5 trials
        direction="minimize", # Since we use MSE
        n_jobs_splits=2, # Use 2 threads to evaluate splits in parallel for each trial
        upload_to_cloud=False, # Set to True to test cloud upload
        **hyperparams_config_example
    )

    print("Starting optimization with parallel splits evaluation...")
    study_results = optimizer.optimize(n_trials=10) # Run 10 trials

    print(f"\\nExample finished. Results are in {results_file}")
    if optimizer.upload_to_cloud:
        print(f"Results also attempted to upload to {optimizer.cloud_name} bucket {optimizer.cloud_bucket}")
    else:
        print("Cloud upload was disabled for this example run.")

# %%
