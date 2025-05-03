import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import numpy as np
import json
import os
from gcp_library.optimization_utils.optimizer_time_series import TimeSeriesOpt

class TestTimeSeriesOptRolling(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(20, 3)
        self.y = np.random.rand(20)
        self.mock_model_cls = MagicMock()
        self.mock_model_instance = MagicMock()
        self.mock_model_cls.return_value = self.mock_model_instance
        self.mock_model_instance.predict.return_value = np.random.rand(5) # Assuming test_size=5

        self.mock_metric = MagicMock(return_value=0.5)

        self.save_path = "rolling_results.jsonl"
        self.s3_bucket = "test-bucket"
        self.s3_key = "test/key.jsonl"
        self.params = {
            'param_float': (0.1, 1.0, "float"),
            'param_int': (1, 10, "int"),
            'param_cat': (['a', 'b'], None, "cat"),
            'param_stat': (5, None, "stationary")
        }
        self.n_splits = 2
        self.test_size = 5
        self.upload_rate = 2
        self.n_trials_per_split = 3

        # Mock Optuna trial and study
        self.mock_trial = MagicMock()
        self.mock_trial.suggest_float.return_value = 0.5
        self.mock_trial.suggest_int.return_value = 5
        self.mock_trial.suggest_categorical.return_value = 'a'
        self.mock_trial.number = 0

        self.mock_study = MagicMock()
        self.mock_study.optimize.side_effect = self.run_objective_multiple_times

    def run_objective_multiple_times(self, objective_func, n_trials, n_jobs):
        """Helper to simulate study.optimize calling the objective."""
        for i in range(n_trials):
            self.mock_trial.number = i
            objective_func(self.mock_trial)

    @patch('gcp_library.optimization_utils.optimizer_time_series.optuna.create_study')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gcp_library.optimization_utils.optimizer_time_series.boto3.client')
    def test_optimize_splits(self, mock_s3_client, mock_file, mock_create_study):
        """Test the optimize_splits method including objective calls and uploads."""
        mock_create_study.return_value = self.mock_study
        mock_s3_instance = MagicMock()
        mock_s3_client.return_value = mock_s3_instance

        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.s3_bucket, self.s3_key,
            n_splits=self.n_splits, test_size=self.test_size,
            upload_s3_rate=self.upload_rate, **self.params
        )

        optimizer.optimize_splits(n_trials=self.n_trials_per_split)

        # Assertions
        self.assertEqual(mock_create_study.call_count, self.n_splits) # Study created per split
        self.assertEqual(self.mock_study.optimize.call_count, self.n_splits) # Optimize called per split

        total_objective_calls = self.n_splits * self.n_trials_per_split
        self.assertEqual(self.mock_model_cls.call_count, total_objective_calls)
        self.assertEqual(self.mock_model_instance.fit.call_count, total_objective_calls)
        self.assertEqual(self.mock_model_instance.predict.call_count, total_objective_calls)
        self.assertEqual(self.mock_metric.call_count, total_objective_calls)

        # Check file writing (called once per trial per split)
        self.assertEqual(mock_file.call_count, self.n_splits) # open called once per split
        self.assertEqual(mock_file().write.call_count, total_objective_calls)

        # Check file paths used for saving
        expected_save_path_split_0 = "rolling_results_0.jsonl"
        expected_save_path_split_1 = "rolling_results_1.jsonl"
        open_calls = mock_file.call_args_list
        self.assertEqual(open_calls[0], call(expected_save_path_split_0, 'a'))
        self.assertEqual(open_calls[1], call(expected_save_path_split_1, 'a'))

        # Check S3 upload trigger (called when trial number + 1 % rate == 0)
        # Trials per split: 0, 1, 2. Upload triggered after trial 1 (index 1).
        expected_uploads = self.n_splits * (self.n_trials_per_split // self.upload_rate)
        self.assertEqual(mock_s3_instance.upload_file.call_count, expected_uploads)

        # Check S3 upload arguments for the last triggered upload (split 1, trial 1)
        expected_s3_save_path = "rolling_results_1.jsonl"
        expected_s3_key = "test/key_split_1.jsonl"
        mock_s3_instance.upload_file.assert_called_with(expected_s3_save_path, self.s3_bucket, expected_s3_key)

    @patch('gcp_library.optimization_utils.optimizer_time_series.boto3.client')
    def test_upload_to_s3(self, mock_s3_client):
        """Test the S3 upload method directly."""
        mock_s3_instance = MagicMock()
        mock_s3_client.return_value = mock_s3_instance
        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.s3_bucket, self.s3_key,
            **self.params
        )
        optimizer.split_number = 1 # Set a split number for testing

        # Mock the existence of the split-specific file
        with patch('os.path.exists', return_value=True):
             # Need to patch os.path.splitext if it's used inside upload_to_s3, but it seems not
             optimizer.upload_to_s3()

        expected_save_path = "rolling_results_1.jsonl"
        expected_s3_key = "test/key_split_1.jsonl"
        mock_s3_client.assert_called_once_with('s3')
        mock_s3_instance.upload_file.assert_called_once_with(expected_save_path, self.s3_bucket, expected_s3_key)

if __name__ == '__main__':
    unittest.main()
