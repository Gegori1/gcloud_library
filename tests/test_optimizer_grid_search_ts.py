import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import numpy as np
import json
import os
import time
from gcp_library.optimization_utils.optimizer_grid_search_ts import TimeSeriesOpt
import optuna # Import optuna for exceptions

class TestTimeSeriesOptGridSearch(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(30, 3)
        self.y = np.random.rand(30)
        self.mock_model_cls = MagicMock()
        self.mock_model_instance = MagicMock()
        self.mock_model_cls.return_value = self.mock_model_instance
        # Simulate predict returning different values for different calls if needed
        self.mock_model_instance.predict.side_effect = [np.random.rand(5), np.random.rand(5), np.random.rand(5)] * 10 # Adjust size and count as needed
        self.mock_model_instance.status = 'optimal' # Default status

        self.mock_metric = MagicMock(return_value=0.5)

        self.param_config = {
            'alpha': (0.1, 1.0, 2, 'float'), # 2 values: 0.1, 1.0
            'beta': (1, 3, 3, 'int'),      # 3 values: 1, 2, 3
            'gamma': (['a', 'b'], None, None, 'cat'), # 2 values: 'a', 'b'
            'delta': (100, None, None, 'stationary') # Stationary
        }
        # Total combinations = 2 * 3 * 2 = 12

        self.save_path = "grid_search_results.jsonl"
        self.cloud_name_gcs = "gcp"
        self.cloud_bucket = "test-bucket"
        self.cloud_key = "test/grid_key.jsonl"
        self.n_splits = 3
        self.test_size = 5
        self.upload_rate = 5 # Upload every 5 trials
        self.n_jobs = 1 # Easier to test sequentially

        # Mock Optuna study, trial, sampler
        self.mock_trial = MagicMock()
        self.mock_trial.number = 0
        # We need to simulate suggest_categorical based on the grid
        self.grid_points = [
            {'alpha': 0.1, 'beta': 1, 'gamma': 'a'}, {'alpha': 0.1, 'beta': 1, 'gamma': 'b'},
            {'alpha': 0.1, 'beta': 2, 'gamma': 'a'}, {'alpha': 0.1, 'beta': 2, 'gamma': 'b'},
            {'alpha': 0.1, 'beta': 3, 'gamma': 'a'}, {'alpha': 0.1, 'beta': 3, 'gamma': 'b'},
            {'alpha': 1.0, 'beta': 1, 'gamma': 'a'}, {'alpha': 1.0, 'beta': 1, 'gamma': 'b'},
            {'alpha': 1.0, 'beta': 2, 'gamma': 'a'}, {'alpha': 1.0, 'beta': 2, 'gamma': 'b'},
            {'alpha': 1.0, 'beta': 3, 'gamma': 'a'}, {'alpha': 1.0, 'beta': 3, 'gamma': 'b'},
        ]
        self.suggest_call_count = 0

        def mock_suggest_categorical(name, choices):
            # Simulate GridSampler suggesting the next point
            current_point = self.grid_points[self.mock_trial.number]
            return current_point[name]

        self.mock_trial.suggest_categorical.side_effect = mock_suggest_categorical

        self.mock_study = MagicMock()
        self.mock_study.optimize.side_effect = self.run_objective_grid
        self.mock_study.best_trial = MagicMock(value=0.4, params={'alpha': 0.1, 'beta': 1, 'gamma': 'a'}) # Example best

        self.mock_sampler = MagicMock(spec=optuna.samplers.GridSampler)

        # Mock os.path.exists for directory check
        self.mock_os_path_exists = patch('os.path.exists', return_value=True)
        self.mock_os_makedirs = patch('os.makedirs')

    def run_objective_grid(self, objective_func, n_trials, n_jobs, catch, show_progress_bar):
        """Helper to simulate study.optimize calling the objective for the grid."""
        self.assertEqual(n_trials, len(self.grid_points)) # Ensure optimize is called with correct total trials
        for i in range(n_trials):
            self.mock_trial.number = i
            try:
                objective_func(self.mock_trial)
            except optuna.exceptions.TrialPruned:
                print(f"Trial {i} pruned.") # Simulate pruning message
            except Exception as e:
                 # If catch=(Exception,) is used, Optuna handles this internally
                 print(f"Trial {i} failed with {e}")


    def setUpOptimizer(self, cloud_name="gcp"):
        """Creates an optimizer instance for testing."""
        return TimeSeriesOpt(
            X=self.X, y=self.y, model=self.mock_model_cls, metric=self.mock_metric,
            param_config=self.param_config, save_path=self.save_path,
            cloud_name=cloud_name, cloud_bucket=self.cloud_bucket, cloud_key=self.cloud_key,
            n_splits=self.n_splits, test_size=self.test_size,
            upload_cloud_rate=self.upload_rate, n_jobs=self.n_jobs
        )

    def test_generate_optuna_grid_search_space(self):
        """Test the generation of the search space for GridSampler."""
        optimizer = self.setUpOptimizer()
        search_space = optimizer._generate_optuna_grid_search_space()

        expected_search_space = {
            'alpha': [0.1, 1.0],
            'beta': [1, 2, 3],
            'gamma': ['a', 'b']
        }
        expected_stationary_params = {'delta': 100}

        self.assertEqual(search_space, expected_search_space)
        self.assertEqual(optimizer.stationary_params, expected_stationary_params)

    @patch('gcp_library.optimization_utils.optimizer_grid_search_ts.optuna.samplers.GridSampler')
    @patch('gcp_library.optimization_utils.optimizer_grid_search_ts.optuna.create_study')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gcp_library.optimization_utils.optimizer_grid_search_ts.storage.Client') # Mock GCS
    @patch('gcp_library.optimization_utils.optimizer_grid_search_ts.ThreadPoolExecutor')
    @patch('time.time') # Mock time
    def test_optimize_splits_gcs(self, mock_time, mock_executor_cls, mock_gcs_client, mock_file, mock_create_study, mock_grid_sampler_cls):
        """Test the main optimize_splits workflow with GCS uploads."""
        # Mock time to control computation_time
        mock_time.side_effect = [10.0, 10.5] * (len(self.grid_points) * self.n_splits) # 0.5s per split fit/predict

        mock_grid_sampler_cls.return_value = self.mock_sampler
        mock_create_study.return_value = self.mock_study
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance
        mock_submit = MagicMock()
        mock_executor_instance.submit.return_value = mock_submit

        # Start mocks for directory check
        self.mock_os_path_exists.start()
        self.mock_os_makedirs.start()

        optimizer = self.setUpOptimizer(cloud_name=self.cloud_name_gcs)
        optimizer.optimize_splits()

        # Stop mocks
        self.mock_os_path_exists.stop()
        self.mock_os_makedirs.stop()

        # Assertions
        mock_grid_sampler_cls.assert_called_once() # Sampler created
        mock_create_study.assert_called_once_with(sampler=self.mock_sampler, direction='minimize')
        self.mock_study.optimize.assert_called_once() # Optimize called once

        total_trials = len(self.grid_points)
        total_model_calls = total_trials * self.n_splits
        self.assertEqual(self.mock_model_cls.call_count, total_model_calls)
        self.assertEqual(self.mock_model_instance.fit.call_count, total_model_calls)
        self.assertEqual(self.mock_model_instance.predict.call_count, total_model_calls)
        self.assertEqual(self.mock_metric.call_count, total_model_calls)

        # Check file writing (called once per trial)
        self.assertEqual(mock_file().write.call_count, total_trials)

        # Check content of the last file write (trial 11)
        expected_params_save = {'alpha': 1.0, 'beta': 3, 'gamma': 'b'} # Only varying params
        expected_computation_time = 0.5 * self.n_splits # 0.5s per split * n_splits
        expected_json_dump = json.dumps({
            "trial": total_trials - 1,
            "target": 0.5, # Mean score
            "params": expected_params_save,
            "computation_time_seconds": expected_computation_time,
            "non_optimal_fits": 0 # Default status was optimal
        }) + '\n'
        mock_file().write.assert_called_with(expected_json_dump)

        # Check GCS upload trigger (called when trial number + 1 % rate == 0)
        # Trials: 0..11. Rate=5. Uploads after trial 4 and 9.
        expected_uploads = total_trials // self.upload_rate
        self.assertEqual(mock_executor_instance.submit.call_count, expected_uploads)
        mock_executor_instance.submit.assert_called_with(optimizer.upload_to_gstorage) # Check last call

        # Check final wait for uploads
        mock_executor_instance.submit().result.assert_called() # Check if result() was called on the future

    @patch('gcp_library.optimization_utils.optimizer_grid_search_ts.optuna.samplers.GridSampler')
    @patch('gcp_library.optimization_utils.optimizer_grid_search_ts.optuna.create_study')
    @patch('builtins.open', new_callable=mock_open)
    @patch('time.time')
    def test_non_optimal_fit_counting(self, mock_time, mock_file, mock_create_study, mock_grid_sampler_cls):
        """Test that non-optimal fits are counted and saved."""
        mock_time.side_effect = [10.0, 10.5] * (len(self.grid_points) * self.n_splits)
        mock_grid_sampler_cls.return_value = self.mock_sampler
        mock_create_study.return_value = self.mock_study

        # Simulate non-optimal status for the second split of the first trial
        def fit_side_effect(*args, **kwargs):
            if self.mock_trial.number == 0 and self.mock_model_instance.fit.call_count == 2: # 2nd call in trial 0
                 self.mock_model_instance.status = 'non-optimal'
            else:
                 self.mock_model_instance.status = 'optimal'
            return None # fit returns None

        self.mock_model_instance.fit.side_effect = fit_side_effect

        self.mock_os_path_exists.start()
        self.mock_os_makedirs.start()

        optimizer = self.setUpOptimizer()
        optimizer.optimize_splits()

        self.mock_os_path_exists.stop()
        self.mock_os_makedirs.stop()

        # Check the first file write (trial 0)
        first_write_call = mock_file().write.call_args_list[0]
        saved_data = json.loads(first_write_call[0][0]) # Get the string arg from the call

        self.assertEqual(saved_data['trial'], 0)
        self.assertEqual(saved_data['non_optimal_fits'], 1) # Should have counted one non-optimal fit

        # Check a later file write (e.g., trial 1) to ensure count reset
        second_write_call = mock_file().write.call_args_list[1]
        saved_data_trial_1 = json.loads(second_write_call[0][0])
        self.assertEqual(saved_data_trial_1['trial'], 1)
        self.assertEqual(saved_data_trial_1['non_optimal_fits'], 0) # Should be 0 for this trial

if __name__ == '__main__':
    unittest.main()
