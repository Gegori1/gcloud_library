import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import json
import os
from gcp_library.optimization_utils.optimizer_1_partition import TimeSeriesOpt
from sklearn.model_selection import train_test_split # Import for patching

class TestTimeSeriesOptOnePartition(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(20, 3)
        self.y = np.random.rand(20)
        self.mock_model_cls = MagicMock()
        self.mock_model_instance = MagicMock()
        self.mock_model_cls.return_value = self.mock_model_instance
        self.mock_model_instance.predict.return_value = np.random.rand(6) # Assuming test_size=0.3 -> 6

        self.mock_metric = MagicMock(return_value=0.5)

        self.save_path = "partition_results.jsonl"
        self.s3_bucket = "test-bucket"
        self.s3_key = "test/partition_key.jsonl"
        self.params = {
            'param_float': (0.1, 1.0, "float"),
            'param_int': (1, 10, "int"),
            'param_cat': (['a', 'b'], None, "cat"),
            'param_stat': (5, None, "stationary")
        }
        self.test_size = 0.3
        self.upload_rate = 2 # Using a small rate for testing upload trigger

        # Mock Optuna trial
        self.mock_trial = MagicMock()
        self.mock_trial.suggest_float.return_value = 0.5
        self.mock_trial.suggest_int.return_value = 5
        self.mock_trial.suggest_categorical.return_value = 'a'
        self.mock_trial.number = 0

        # Mock train_test_split return values
        self.X_train_mock = self.X[:14]
        self.X_val_mock = self.X[14:]
        self.y_train_mock = self.y[:14]
        self.y_val_mock = self.y[14:]

    @patch('gcp_library.optimization_utils.optimizer_1_partition.train_test_split')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gcp_library.optimization_utils.optimizer_1_partition.boto3.client')
    @patch('os.path.exists', return_value=True) # Assume file exists for upload
    def test_call_and_s3_upload(self, mock_os_exists, mock_s3_client, mock_file, mock_train_test_split):
        """Test the __call__ method and S3 upload trigger."""
        mock_train_test_split.return_value = (self.X_train_mock, self.X_val_mock, self.y_train_mock, self.y_val_mock)
        mock_s3_instance = MagicMock()
        mock_s3_client.return_value = mock_s3_instance

        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.s3_bucket, self.s3_key,
            test_size=self.test_size, **self.params
        )

        # Simulate two trials to trigger upload (rate is 2)
        self.mock_trial.number = 0
        result1 = optimizer(self.mock_trial)
        self.mock_trial.number = 1
        result2 = optimizer(self.mock_trial)

        # Assertions
        self.assertEqual(result1, 0.5)
        self.assertEqual(result2, 0.5)

        # Check train_test_split call
        mock_train_test_split.assert_called_with(self.X, self.y, test_size=self.test_size, shuffle=False)
        self.assertEqual(mock_train_test_split.call_count, 2) # Called per trial

        # Check model fitting and prediction
        self.assertEqual(self.mock_model_cls.call_count, 2)
        self.assertEqual(self.mock_model_instance.fit.call_count, 2)
        # Ensure fit was called with the correct split data in the last call
        np.testing.assert_array_equal(self.mock_model_instance.fit.call_args[0][0], self.X_train_mock)
        np.testing.assert_array_equal(self.mock_model_instance.fit.call_args[0][1], self.y_train_mock)

        self.assertEqual(self.mock_model_instance.predict.call_count, 2)
        # Ensure predict was called with the correct split data in the last call
        np.testing.assert_array_equal(self.mock_model_instance.predict.call_args[0][0], self.X_val_mock)

        self.assertEqual(self.mock_metric.call_count, 2)
        # Ensure metric was called with the correct split data in the last call
        np.testing.assert_array_equal(self.mock_metric.call_args[0][0], self.y_val_mock)
        # self.assertEqual(self.mock_metric.call_args[0][1].shape, self.y_val_mock.shape) # Check prediction shape if needed

        # Check parameter suggestions
        self.mock_trial.suggest_float.assert_called_with('param_float', 0.1, 1.0)
        self.mock_trial.suggest_int.assert_called_with('param_int', 1, 10)
        self.mock_trial.suggest_categorical.assert_called_with('param_cat', ['a', 'b'])

        # Check file writing
        self.assertEqual(mock_file().write.call_count, 2) # Called once per trial
        expected_params_save = {'param_float': 0.5, 'param_int': 5, 'param_cat': 'a'} # Stationary not saved
        expected_json_dump = json.dumps({"target": 0.5, "params": expected_params_save}) + '\n'
        mock_file().write.assert_called_with(expected_json_dump) # Check last call

        # Check S3 upload trigger (called when trial number + 1 % rate == 0)
        # Rate is 2, called after trial 1 (index 1)
        self.assertEqual(mock_s3_instance.upload_file.call_count, 1)
        mock_s3_instance.upload_file.assert_called_once_with(self.save_path, self.s3_bucket, self.s3_key)
        mock_os_exists.assert_called_with(self.save_path) # Check os.path.exists was called

    @patch('gcp_library.optimization_utils.optimizer_1_partition.boto3.client')
    @patch('os.path.exists', return_value=True)
    def test_upload_to_s3_exists(self, mock_os_exists, mock_s3_client):
        """Test the S3 upload method when file exists."""
        mock_s3_instance = MagicMock()
        mock_s3_client.return_value = mock_s3_instance
        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.s3_bucket, self.s3_key,
            **self.params
        )
        optimizer.upload_to_s3()
        mock_os_exists.assert_called_once_with(self.save_path)
        mock_s3_client.assert_called_once_with('s3')
        mock_s3_instance.upload_file.assert_called_once_with(self.save_path, self.s3_bucket, self.s3_key)

    @patch('gcp_library.optimization_utils.optimizer_1_partition.boto3.client')
    @patch('os.path.exists', return_value=False)
    def test_upload_to_s3_not_exists(self, mock_os_exists, mock_s3_client):
        """Test the S3 upload method when file does not exist."""
        mock_s3_instance = MagicMock()
        mock_s3_client.return_value = mock_s3_instance
        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.s3_bucket, self.s3_key,
            **self.params
        )
        optimizer.upload_to_s3()
        mock_os_exists.assert_called_once_with(self.save_path)
        mock_s3_client.assert_called_once_with('s3')
        mock_s3_instance.upload_file.assert_not_called() # Should not upload if file doesn't exist

if __name__ == '__main__':
    unittest.main()
