import os
import json
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Attempt to import cloud libraries, but make them optional for core functionality
try:
    from google.cloud import storage
except ImportError:
    storage = None
try:
    import boto3
except ImportError:
    boto3 = None

# Example imports for __main__
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


class SequentialSimulator:
    """
    Class to run a series of simulations with specified parameters without optimization.
    It accepts parameter configurations where non-stationary parameters are defined by lists
    of values. It iterates through these lists, running one simulation for each corresponding
    set of parameter values. All lists for sequential parameters must have the same length.
    Stationary parameters remain constant across all simulations.

    Results are saved locally and can be optionally uploaded to cloud storage.

    param_config format:
    {
        'param_name1': ([val1_run1, val1_run2, ...], 'sequential'),
        'param_name2': ([val2_run1, val2_run2, ...], 'sequential'), # Must be same length as param_name1's list
        'param_name_stat': (fixed_value, 'stationary')
    }
    """
    def __init__(self, X_train, y_train, X_val, y_val, model, metric,
                 param_config: dict,
                 save_path: str,
                 cloud_name: str = "", cloud_bucket: str = "", cloud_key: str = "",
                 upload_cloud_rate=50,
                 upload_to_cloud=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_cls = model
        self.metric = metric
        self.param_config = param_config
        self.save_path = save_path
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

        print("Preparing simulation runs and identifying parameters:")
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
            self.num_simulations = 1 if self.stationary_params else 0 # Run once if only stationary, 0 if no params
            print("No sequential parameters found." + ( " Will run 1 simulation with stationary parameters." if self.stationary_params else " No parameters to run."))
        else:
            self.num_simulations = sequential_list_length
            if self.num_simulations == 0: # Should be caught by empty list check
                 raise ValueError("Sequential parameter lists are effectively empty, resulting in 0 simulations.")
        
        if self.num_simulations > 0:
            print(f"Total simulations to run: {self.num_simulations}")


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
            # print("Cloud upload is disabled. Skipping upload.") # Too verbose for periodic checks
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
            # print("Cloud upload is disabled or no pending uploads. Skipping wait.")
            return

        print(f"Waiting for {len(self.upload_futures)} upload(s) to complete...")
        for future in as_completed(self.upload_futures):
            try:
                future.result() # Wait for the upload to finish and retrieve result/exception
            except Exception as e:
                print(f"An upload task failed: {e}")
        self.upload_futures = [] # Reset for next batch or final
        print("All pending uploads finished.")


    def save_result(self, simulation_idx, metric_value, params, computation_time, non_optimal_count=0):
        metric_value_to_save = metric_value
        if isinstance(metric_value, float):
            if math.isnan(metric_value):
                metric_value_to_save = "NaN"
            elif math.isinf(metric_value):
                metric_value_to_save = "Infinity" if metric_value > 0 else "-Infinity"
        elif not isinstance(metric_value, (int, str, type(None))):
            metric_value_to_save = str(metric_value)

        serializable_params = {
            k: (str(v) if isinstance(v, type) else v) # Convert type objects to string
            for k, v in params.items()
            if isinstance(v, (int, float, str, bool, type(None), type))
        }
        serializable_params = { # Second pass for lists/tuples if any (should not happen with current param construction)
            k: (list(v) if isinstance(v, tuple) else v)
            for k,v in serializable_params.items()
        }


        result_data = {
            "simulation_run": simulation_idx + 1, # 1-indexed for reporting
            "target": metric_value_to_save,
            "params": serializable_params,
            "computation_time_seconds": round(computation_time, 4),
            "non_optimal_fits": non_optimal_count
        }
        try:
            with open(self.save_path, 'a') as f:
                json.dump(result_data, f)
                f.write('\\n')
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

        print(f"--- Starting Sequential Simulations ({self.num_simulations} total) ---")
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
            
            # Handle case where only stationary params exist
            if not self.sequential_param_lists and self.stationary_params and i==0:
                 print(f"\nRunning simulation {i+1}/{self.num_simulations} with stationary params: {current_params}")
            elif self.sequential_param_lists : # Standard case with sequential params
                 print(f"\nRunning simulation {i+1}/{self.num_simulations} with sequential params: {sequential_params_for_this_run}")
                 if self.stationary_params: print(f"  and stationary params: {self.stationary_params}")


            metric_val = None
            non_optimal_count = 0
            start_time = time.time()

            try:
                model_instance = self.model_cls(**current_params)
                model_instance.fit(self.X_train, self.y_train)

                if hasattr(model_instance, 'status') and model_instance.status != 'optimal':
                    non_optimal_count += 1
                    print(f"    Warning: Model status '{model_instance.status}' for params {current_params}.")

                y_pred = model_instance.predict(self.X_val)
                if hasattr(y_pred, 'flatten'):
                    y_pred = y_pred.flatten()

                calculated_metric = self.metric(self.y_val, y_pred)

                if not isinstance(calculated_metric, (int, float)) or math.isnan(calculated_metric) or math.isinf(calculated_metric):
                    print(f"    Warning: Invalid metric value ({calculated_metric}) for params {current_params}. Recording as 'InvalidMetric'.")
                    metric_val = "InvalidMetric"
                else:
                    metric_val = float(calculated_metric)

            except Exception as e:
                print(f"    Error during model evaluation with params {current_params}: {e}")
                metric_val = "EvaluationError"

            end_time = time.time()
            computation_time = end_time - start_time

            self.save_result(i, metric_val, current_params, computation_time, non_optimal_count)
            print(f"  Simulation {i+1} finished. Metric: {metric_val}, Time: {computation_time:.2f}s")


            if self.upload_to_cloud and (i + 1) % self.upload_cloud_rate == 0 and (i + 1) < self.num_simulations:
                print(f"\nSimulation {i + 1}: Reached upload threshold ({self.upload_cloud_rate}). Triggering upload...")
                self._trigger_upload()
                self._wait_for_uploads() # Wait for periodic uploads to avoid too many concurrent threads / memory issues

        print("\n--- Finished All Simulations ---")
        if self.upload_to_cloud:
            print("Triggering final upload of results...")
            self._trigger_upload()
            self._wait_for_uploads()
            print("All uploads finished.")
        else:
            print("Cloud upload was disabled for this run.")
        
        print(f"All simulation results saved to {self.save_path}")


if __name__ == "__main__":
    # Generate sample data
    X_data = np.random.randn(100, 5)
    y_data = X_data[:, 0] * 2.5 + X_data[:, 1] * -1.5 + np.random.randn(100) * 1.2

    X_train_ex, X_val_ex, y_train_ex, y_val_ex = train_test_split(
        X_data, y_data, test_size=0.25, shuffle=True, random_state=42
    )

    def example_metric_mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    # Define parameter configuration for sequential simulation
    param_config_example = {
        'alpha': ([0.1, 1.0, 5.0, 10.0], 'sequential'),
        'solver': (['svd', 'cholesky', 'lsqr', 'sag'], 'sequential'), # Must be same length as alpha
        'fit_intercept': (True, 'stationary'),
        'tol': ([1e-4, 1e-3, 1e-2, 1e-1], 'sequential') # Must be same length
    }
    # This will run 4 simulations:
    # 1. alpha=0.1, solver='svd', fit_intercept=True, tol=1e-4
    # 2. alpha=1.0, solver='cholesky', fit_intercept=True, tol=1e-3
    # 3. alpha=5.0, solver='lsqr', fit_intercept=True, tol=1e-2
    # 4. alpha=10.0, solver='sag', fit_intercept=True, tol=1e-1

    results_file_path = "sequential_simulation_results.jsonl"
    
    # Cloud configuration (optional, replace with your details if testing)
    cloud_provider_name = "gcp"  # or "aws", or "" to skip
    cloud_storage_bucket = "your-gcp-bucket-name" 
    cloud_storage_key = "simulation_tests/sequential_results.jsonl"

    # Clean up previous results file if it exists
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
        print(f"Removed previous results file: {results_file_path}")

    upload_frequency = 2 # Upload results to cloud every 2 simulations

    simulator_instance = SequentialSimulator(
        X_train=X_train_ex,
        y_train=y_train_ex,
        X_val=X_val_ex,
        y_val=y_val_ex,
        model=Ridge, # Scikit-learn model class
        metric=example_metric_mse,
        param_config=param_config_example,
        save_path=results_file_path,
        cloud_name=cloud_provider_name,
        cloud_bucket=cloud_storage_bucket,
        cloud_key=cloud_storage_key,
        upload_cloud_rate=upload_frequency,
        upload_to_cloud=False # Set to True to test actual cloud uploads (ensure credentials and libraries)
    )

    simulator_instance.run_simulations()

    print(f"\n--- Sequential Simulation Example Complete ---")
    print(f"All simulation results potentially saved to {results_file_path}")
    if simulator_instance.upload_to_cloud and cloud_provider_name and cloud_storage_bucket:
        print(f"Results also attempted to be uploaded to {cloud_provider_name} bucket '{cloud_storage_bucket}' at '{cloud_storage_key}'")
    elif simulator_instance.upload_to_cloud:
        print("Cloud upload was enabled but might have been skipped due to missing config or libraries.")
    else:
        print("Cloud upload was disabled for this example run.")

    # Example: Run with only stationary parameters
    print("\n--- Example: Running with only stationary parameters ---")
    param_config_stationary_only = {
        'alpha': (1.0, 'stationary'),
        'solver': ('auto', 'stationary'),
        'fit_intercept': (False, 'stationary')
    }
    results_file_stationary_path = "sequential_simulation_stationary_only_results.jsonl"
    if os.path.exists(results_file_stationary_path):
        os.remove(results_file_stationary_path)

    simulator_stationary_only = SequentialSimulator(
        X_train=X_train_ex, y_train=y_train_ex, X_val=X_val_ex, y_val=y_val_ex,
        model=Ridge, metric=example_metric_mse,
        param_config=param_config_stationary_only,
        save_path=results_file_stationary_path,
        upload_to_cloud=False
    )
    simulator_stationary_only.run_simulations()
    print(f"Stationary only simulation results saved to {results_file_stationary_path}")

    # Example: Run with no parameters (should do nothing or print a message)
    print("\n--- Example: Running with no parameters ---")
    param_config_empty = {}
    results_file_empty_path = "sequential_simulation_empty_params_results.jsonl"
    if os.path.exists(results_file_empty_path):
        os.remove(results_file_empty_path)
    simulator_empty = SequentialSimulator(
        X_train=X_train_ex, y_train=y_train_ex, X_val=X_val_ex, y_val=y_val_ex,
        model=Ridge, metric=example_metric_mse,
        param_config=param_config_empty,
        save_path=results_file_empty_path,
        upload_to_cloud=False
    )
    simulator_empty.run_simulations()
    # Check if results_file_empty_path was created (it shouldn't be if no simulations run)
    if os.path.exists(results_file_empty_path):
         print(f"Empty param simulation results file created at {results_file_empty_path} (unexpected for 0 runs).")
    else:
         print(f"Empty param simulation correctly resulted in no output file at {results_file_empty_path}.")
