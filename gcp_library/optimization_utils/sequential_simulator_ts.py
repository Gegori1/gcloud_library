import os
import json
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Attempt to import cloud libraries, but make them optional
try:
    from google.cloud import storage
except ImportError:
    storage = None
try:
    import boto3
except ImportError:
    boto3 = None

# Example imports for __main__ (will be used if script is run directly)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


class SequentialSimulatorTimeSeries:
    """
    Class to run a series of simulations with specified parameters for time-series data.
    It accepts parameter configurations where non-stationary parameters are defined by lists
    of values. It iterates through these lists, running one simulation for each corresponding
    set of parameter values. All lists for sequential parameters must have the same length.
    Stationary parameters remain constant across all simulations.

    For each parameter set, the model is evaluated across multiple time-series splits,
    and the mean metric score across these splits is recorded.

    Results are saved locally and can be optionally uploaded to cloud storage.

    param_config format:
    {
        'param_name1': ([val1_run1, val1_run2, ...], 'sequential'),
        'param_name2': ([val2_run1, val2_run2, ...], 'sequential'), # Must be same length
        'param_name_stat': (fixed_value, 'stationary')
    }
    """
    def __init__(self, X, y, model, metric,
                 param_config: dict,
                 save_path: str,
                 n_splits=5, test_size=None, gap=0,
                 cloud_name: str = "", cloud_bucket: str = "", cloud_key: str = "",
                 upload_cloud_rate=50,
                 upload_to_cloud=False):
        self.X = X
        self.y = y
        self.model_cls = model
        self.metric = metric
        self.param_config = param_config
        self.save_path = save_path

        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size, gap=self.gap)
        # Store actual split indices once, if X and y are not expected to change
        # For very large datasets, splitting on the fly might be preferred if memory is a concern
        # For now, let's assume we can pre-calculate or do it on the fly in the loop.
        # self.tscv_indices = list(self.tscv.split(self.X, self.y))


        self.cloud_name = cloud_name
        self.cloud_bucket = cloud_bucket
        self.cloud_key = cloud_key
        self.upload_cloud_rate = upload_cloud_rate
        self.upload_to_cloud = upload_to_cloud

        if self.upload_to_cloud:
            self.upload_executor = ThreadPoolExecutor(max_workers=4)
            self.upload_futures = []
            if self.cloud_name.lower() in ["google", "gcp"] and storage is None:
                print("Warning: google-cloud-storage library not found. GCS upload will fail.")
            if self.cloud_name.lower() in ["amazon", "aws"] and boto3 is None:
                print("Warning: boto3 library not found. S3 upload will fail.")

        self.stationary_params = {}
        self.sequential_param_lists = {}
        self.num_simulations = 0

        self._ensure_save_dir()

    def _ensure_save_dir(self):
        save_dir = os.path.dirname(self.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def _prepare_simulation_runs(self):
        self.stationary_params = {}
        self.sequential_param_lists = {}
        sequential_list_length = -1

        print("Preparing time-series simulation runs and identifying parameters:")
        for name, config_entry in self.param_config.items():
            if not isinstance(config_entry, tuple) or len(config_entry) != 2:
                raise ValueError(
                    f"Parameter config for '{name}' must be a 2-element tuple: (values_list_or_value, type_str)"
                )

            values_or_value, ptype = config_entry
            ptype = ptype.lower()

            if ptype == 'sequential':
                if not isinstance(values_or_value, (list, tuple)):
                    raise ValueError(f"Values for sequential param '{name}' must be a list or tuple.")
                if not values_or_value:
                    raise ValueError(f"Values list for sequential param '{name}' cannot be empty.")

                current_list_length = len(values_or_value)
                if sequential_list_length == -1:
                    sequential_list_length = current_list_length
                elif sequential_list_length != current_list_length:
                    raise ValueError(
                        f"All 'sequential' parameter lists must have the same length. "
                        f"Param '{name}' has length {current_list_length}, expected {sequential_list_length}."
                    )
                self.sequential_param_lists[name] = list(values_or_value)
                print(f"  - {name} (sequential): {len(values_or_value)} values -> {values_or_value}")

            elif ptype == 'stationary':
                self.stationary_params[name] = values_or_value
                print(f"  - {name} (stationary): Fixed value -> {values_or_value}")
            else:
                raise ValueError(f"Unsupported parameter type: '{ptype}' for param '{name}'. Use 'sequential' or 'stationary'.")

        if not self.sequential_param_lists:
            self.num_simulations = 1 if self.stationary_params else 0
            print("No sequential parameters found." + (" Will run 1 simulation with stationary parameters." if self.stationary_params else " No parameters to run."))
        else:
            self.num_simulations = sequential_list_length
            if self.num_simulations == 0:
                 raise ValueError("Sequential parameter lists are effectively empty, resulting in 0 simulations.")
        
        if self.num_simulations > 0:
            print(f"Total simulation runs (parameter sets): {self.num_simulations}")
            print(f"Each run will be evaluated over {self.n_splits} time-series splits.")


    def upload_to_s3(self):
        if boto3 is None:
            print("boto3 library is not installed. Cannot upload to S3.")
            return
        print(f"Attempting to upload {self.save_path} to S3 bucket {self.cloud_bucket}...")
        try:
            s3 = boto3.client('s3')
            s3.upload_file(self.save_path, self.cloud_bucket, self.cloud_key)
            print("S3 upload successful.")
        except Exception as e:
            print(f"S3 upload failed: {e}")

    def upload_to_gstorage(self):
        if storage is None:
            print("google-cloud-storage library is not installed. Cannot upload to GCS.")
            return
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
            return
        if not self.cloud_name or not self.cloud_bucket or not self.cloud_key:
            print("Warning: Cloud configuration (name, bucket, key) is incomplete. Skipping upload.")
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
        if not self.upload_to_cloud or not hasattr(self, 'upload_futures') or not self.upload_futures:
            return
        print(f"Waiting for {len(self.upload_futures)} upload(s) to complete...")
        for future in as_completed(self.upload_futures):
            try:
                future.result()
            except Exception as e:
                print(f"An upload task failed: {e}")
        self.upload_futures = []
        print("All pending uploads finished.")

    def save_result(self, simulation_idx, mean_metric_score, params, computation_time, total_non_optimal_count, individual_scores):
        score_to_save = mean_metric_score
        if isinstance(mean_metric_score, float):
            if math.isnan(mean_metric_score):
                score_to_save = "NaN"
            elif math.isinf(mean_metric_score):
                score_to_save = "Infinity" if mean_metric_score > 0 else "-Infinity"
        elif not isinstance(mean_metric_score, (int, str, type(None))): # Allow None if all splits failed
            score_to_save = str(mean_metric_score)

        serializable_params = {
            k: (str(v) if isinstance(v, type) else v)
            for k, v in params.items()
            if isinstance(v, (int, float, str, bool, type(None), type))
        }
        serializable_params = {k: (list(v) if isinstance(v, tuple) else v) for k,v in serializable_params.items()}
        
        # Serialize individual scores similarly
        processed_individual_scores = []
        for score in individual_scores:
            if isinstance(score, float):
                if math.isnan(score): processed_individual_scores.append("NaN")
                elif math.isinf(score): processed_individual_scores.append("Infinity" if score > 0 else "-Infinity")
                else: processed_individual_scores.append(round(score, 6)) # Keep precision for individual scores
            elif isinstance(score, (int, str)): processed_individual_scores.append(score)
            else: processed_individual_scores.append(str(score))


        result_data = {
            "simulation_run": simulation_idx + 1,
            "mean_metric_score": score_to_save,
            "params": serializable_params,
            "computation_time_seconds": round(computation_time, 4),
            "total_non_optimal_fits_across_splits": total_non_optimal_count,
            "individual_split_scores": processed_individual_scores,
            "num_splits_evaluated": len(individual_scores)
        }
        try:
            with open(self.save_path, 'a') as f:
                json.dump(result_data, f)
                f.write('\n')
        except IOError as e:
            print(f"Error saving simulation result to {self.save_path}: {e}")
        except TypeError as e:
            print(f"Error serializing simulation result {simulation_idx + 1}: {result_data}. Error: {e}")

    def run_simulations(self):
        try:
            self._prepare_simulation_runs()
            if self.num_simulations == 0:
                print("No simulations to run based on the provided parameter configuration. Exiting.")
                return
        except Exception as e:
            print(f"Error preparing simulation runs: {e}")
            return

        print(f"--- Starting Sequential Time-Series Simulations ({self.num_simulations} total parameter sets) ---")
        if self.stationary_params:
            print(f"  Stationary parameters: {self.stationary_params}")
        if self.sequential_param_lists:
            print(f"  Sequential parameters will be iterated.")

        for i in range(self.num_simulations):
            current_params = {**self.stationary_params}
            sequential_params_for_this_run = {}

            if self.sequential_param_lists:
                for name, value_list in self.sequential_param_lists.items():
                    param_val = value_list[i]
                    current_params[name] = param_val
                    sequential_params_for_this_run[name] = param_val
            
            if not self.sequential_param_lists and self.stationary_params and i==0:
                 print(f"\nRunning simulation set {i+1}/{self.num_simulations} with stationary params: {current_params}")
            elif self.sequential_param_lists:
                 print(f"\nRunning simulation set {i+1}/{self.num_simulations} with sequential params: {sequential_params_for_this_run}")
                 if self.stationary_params: print(f"  and stationary params: {self.stationary_params}")

            start_time_param_set = time.time()
            
            split_scores = []
            total_non_optimal_for_param_set = 0
            
            # Ensure X and y are numpy arrays for proper indexing by TimeSeriesSplit
            X_np = np.asarray(self.X)
            y_np = np.asarray(self.y)

            for split_idx, (train_index, test_index) in enumerate(self.tscv.split(X_np, y_np)):
                print(f"  Split {split_idx + 1}/{self.n_splits} for param set {i+1}")
                X_train_split, X_test_split = X_np[train_index], X_np[test_index]
                y_train_split, y_test_split = y_np[train_index], y_np[test_index]
                
                split_metric_val = None
                non_optimal_this_split = 0

                try:
                    model_instance = self.model_cls(**current_params)
                    model_instance.fit(X_train_split, y_train_split)

                    if hasattr(model_instance, 'status') and model_instance.status != 'optimal':
                        non_optimal_this_split += 1
                        print(f"    Warning: Model status '{model_instance.status}' in split {split_idx+1} for params {current_params}.")
                    
                    total_non_optimal_for_param_set += non_optimal_this_split
                    
                    y_pred_split = model_instance.predict(X_test_split)
                    if hasattr(y_pred_split, 'flatten'):
                        y_pred_split = y_pred_split.flatten()

                    calculated_metric = self.metric(y_test_split, y_pred_split)

                    if not isinstance(calculated_metric, (int, float)) or math.isnan(calculated_metric) or math.isinf(calculated_metric):
                        print(f"    Warning: Invalid metric ({calculated_metric}) in split {split_idx+1}. Recording as 'InvalidSplitMetric'.")
                        split_metric_val = "InvalidSplitMetric" # Placeholder for individual score list
                    else:
                        split_metric_val = float(calculated_metric)
                    
                except Exception as e:
                    print(f"    Error during model evaluation in split {split_idx+1} with params {current_params}: {e}")
                    split_metric_val = "SplitEvaluationError" # Placeholder

                split_scores.append(split_metric_val)

            # Calculate mean score for the parameter set
            valid_split_scores = [s for s in split_scores if isinstance(s, (int, float))]
            mean_score_for_param_set = None
            if valid_split_scores:
                mean_score_for_param_set = np.mean(valid_split_scores)
            else: # All splits failed or returned non-numeric
                mean_score_for_param_set = "AllSplitsFailed" 
                print(f"    Warning: All splits failed evaluation for parameter set {i+1}. Mean score recorded as '{mean_score_for_param_set}'.")


            end_time_param_set = time.time()
            computation_time_param_set = end_time_param_set - start_time_param_set

            self.save_result(i, mean_score_for_param_set, current_params, computation_time_param_set, total_non_optimal_for_param_set, split_scores)
            print(f"  Parameter set {i+1} finished. Mean Metric: {mean_score_for_param_set}, Total Time: {computation_time_param_set:.2f}s")

            if self.upload_to_cloud and (i + 1) % self.upload_cloud_rate == 0 and (i + 1) < self.num_simulations:
                print(f"\nParameter set {i + 1}: Reached upload threshold ({self.upload_cloud_rate}). Triggering upload...")
                self._trigger_upload()
                self._wait_for_uploads()

        print("\n--- Finished All Time-Series Simulation Sets ---")
        if self.upload_to_cloud:
            print("Triggering final upload of results...")
            self._trigger_upload()
            self._wait_for_uploads()
            print("All uploads finished.")
        else:
            print("Cloud upload was disabled for this run.")
        
        print(f"All simulation results saved to {self.save_path}")


if __name__ == "__main__":
    # Generate sample time-series data
    n_samples = 150
    X_data_ts = np.random.randn(n_samples, 3)
    # Create a simple time trend and seasonality for y
    time_trend = np.linspace(0, 10, n_samples)
    seasonality = np.sin(np.linspace(0, 3 * np.pi, n_samples)) * 5
    y_data_ts = X_data_ts[:, 0] * 2.0 + X_data_ts[:, 1] * -1.0 + time_trend + seasonality + np.random.randn(n_samples) * 0.5

    def example_metric_mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    param_config_ts_example = {
        'alpha': ([0.5, 1.0, 2.0], 'sequential'),
        'solver': (['svd', 'cholesky', 'lsqr'], 'sequential'), # Must be same length as alpha
        'fit_intercept': (True, 'stationary'),
        'tol': ([1e-3, 1e-2, 1e-1], 'sequential') # Must be same length
    }
    # This will run 3 simulation sets. Each set evaluated over n_splits.
    # 1. alpha=0.5, solver='svd', fit_intercept=True, tol=1e-3 -> mean_mse_over_splits
    # 2. alpha=1.0, solver='cholesky', fit_intercept=True, tol=1e-2 -> mean_mse_over_splits
    # 3. alpha=2.0, solver='lsqr', fit_intercept=True, tol=1e-1 -> mean_mse_over_splits

    results_file_ts_path = "sequential_ts_simulation_results.jsonl"
    
    cloud_provider_name_ts = "gcp"
    cloud_storage_bucket_ts = "your-gcp-bucket-name" # Replace if testing cloud
    cloud_storage_key_ts = "simulation_tests/sequential_ts_results.jsonl" # Replace

    if os.path.exists(results_file_ts_path):
        os.remove(results_file_ts_path)
        print(f"Removed previous results file: {results_file_ts_path}")

    ts_simulator = SequentialSimulatorTimeSeries(
        X=X_data_ts,
        y=y_data_ts,
        model=Ridge,
        metric=example_metric_mse,
        param_config=param_config_ts_example,
        save_path=results_file_ts_path,
        n_splits=4, # Number of time series splits
        test_size=int(n_samples * 0.15), # 15% for test set in each split
        gap=1,
        cloud_name=cloud_provider_name_ts,
        cloud_bucket=cloud_storage_bucket_ts,
        cloud_key=cloud_storage_key_ts,
        upload_cloud_rate=2, # Upload every 2 parameter sets
        upload_to_cloud=False # Set to True to test cloud uploads
    )

    ts_simulator.run_simulations()

    print(f"\n--- Sequential Time-Series Simulation Example Complete ---")
    print(f"Results (mean scores across splits) saved to {results_file_ts_path}")
    if ts_simulator.upload_to_cloud and cloud_provider_name_ts and cloud_storage_bucket_ts:
        print(f"Results also attempted to be uploaded to {cloud_provider_name_ts} bucket '{cloud_storage_bucket_ts}' at '{cloud_storage_key_ts}'")
    else:
        print("Cloud upload was disabled or not fully configured for this example run.")

    # Example: Run with only stationary parameters for time series
    print("\n--- Example: TS Running with only stationary parameters ---")
    param_config_ts_stationary_only = {
        'alpha': (0.8, 'stationary'),
        'fit_intercept': (False, 'stationary')
    }
    results_file_ts_stationary_path = "sequential_ts_sim_stationary_results.jsonl"
    if os.path.exists(results_file_ts_stationary_path):
        os.remove(results_file_ts_stationary_path)

    ts_simulator_stationary = SequentialSimulatorTimeSeries(
        X=X_data_ts, y=y_data_ts, model=Ridge, metric=example_metric_mse,
        param_config=param_config_ts_stationary_only,
        save_path=results_file_ts_stationary_path,
        n_splits=3, test_size=int(n_samples * 0.2),
        upload_to_cloud=False
    )
    ts_simulator_stationary.run_simulations()
    print(f"Stationary only TS simulation results saved to {results_file_ts_stationary_path}")
