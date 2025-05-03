import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import json
from gcp_library.optimization_utils.optimizer_ts_allatonce import TimeSeriesOpt

class TestTimeSeriesOptAllAtOnce(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(20, 3)
        self.y = np.random.rand(20)
        self.mock_model_cls = MagicMock()
        self.mock_model_instance = MagicMock()
        self.mock_model_cls.return_value = self.mock_model_instance
        self.mock_model_instance.predict.return_value = np.random.rand(5) # Assuming test_size=5 in TimeSeriesSplit

        self.mock_metric = MagicMock(return_value=0.5)
        self.mock_metric.__name__ = "mock_metric" # Add name attribute

        self.save_path = "test_results.jsonl"
        self.cloud_name_s3 = "aws"
        self.cloud_name_gcs = "gcp"
        self.cloud_bucket = "test-bucket"
        self.cloud_key = "test/key.jsonl"
        self.params = {
            'param_float': (0.1, 1.0, "float"),
            'param_int': (1, 10, "int"),
            'param_cat': (['a', 'b'], None, "cat"),
            'param_stat': (5, None, "stationary")
        }
        self.n_splits = 3
        self.test_size = 5 # Ensure predict returns array of this size
        self.upload_rate = 2

        # Mock Optuna trial
        self.mock_trial = MagicMock()
        self.mock_trial.suggest_float.return_value = 0.5
        self.mock_trial.suggest_int.return_value = 5
        self.mock_trial.suggest_categorical.return_value = 'a'
        self.mock_trial.number = 0 # Start trial number

    @patch('builtins.open', new_callable=mock_open)
    @patch('gcp_library.optimization_utils.optimizer_ts_allatonce.boto3.client')
    @patch('gcp_library.optimization_utils.optimizer_ts_allatonce.storage.Client')
    @patch('gcp_library.optimization_utils.optimizer_ts_allatonce.ThreadPoolExecutor')
    def test_call_s3_upload(self, mock_executor_cls, mock_gcs_client, mock_s3_client, mock_file):
        """Test the __call__ method with S3 upload trigger."""
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance
        mock_submit = MagicMock()
        mock_executor_instance.submit.return_value = mock_submit

        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.cloud_name_s3, self.cloud_bucket, self.cloud_key,
            n_splits=self.n_splits, test_size=self.test_size, upload_cloud_rate=self.upload_rate, **self.params
        )

        # Simulate two trials to trigger upload
        self.mock_trial.number = 0
        result1 = optimizer(self.mock_trial)
        self.mock_trial.number = 1
        result2 = optimizer(self.mock_trial)

        # Assertions
        self.assertEqual(result1, 0.5) # Mean of scores (mock_metric returns 0.5)
        self.assertEqual(result2, 0.5)
        self.assertEqual(self.mock_model_cls.call_count, self.n_splits * 2) # Called per split per trial
        self.assertEqual(self.mock_model_instance.fit.call_count, self.n_splits * 2)
        self.assertEqual(self.mock_model_instance.predict.call_count, self.n_splits * 2)
        self.assertEqual(self.mock_metric.call_count, self.n_splits * 2)

        # Check parameter suggestions
        self.mock_trial.suggest_float.assert_called_with('param_float', 0.1, 1.0)
        self.mock_trial.suggest_int.assert_called_with('param_int', 1, 10)
        self.mock_trial.suggest_categorical.assert_called_with('param_cat', ['a', 'b'])

        # Check file writing
        self.assertEqual(mock_file().write.call_count, 2) # Called once per trial
        expected_params_save = {'param_float': 0.5, 'param_int': 5, 'param_cat': 'a'} # Stationary not saved
        expected_json_dump = json.dumps({"target": 0.5, "params": expected_params_save}) + '\n'
        mock_file().write.assert_called_with(expected_json_dump) # Check last call

        # Check S3 upload trigger
        self.assertEqual(mock_executor_instance.submit.call_count, 1) # Called once after trial 1 (index 1)
        mock_executor_instance.submit.assert_called_with(optimizer.upload_to_s3)
        mock_s3_client.assert_not_called() # Client itself not called directly in __call__
        mock_gcs_client.assert_not_called()

    @patch('builtins.open', new_callable=mock_open)
    @patch('gcp_library.optimization_utils.optimizer_ts_allatonce.boto3.client')
    @patch('gcp_library.optimization_utils.optimizer_ts_allatonce.storage.Client')
    @patch('gcp_library.optimization_utils.optimizer_ts_allatonce.ThreadPoolExecutor')
    def test_call_gcs_upload(self, mock_executor_cls, mock_gcs_client, mock_s3_client, mock_file):
        """Test the __call__ method with GCS upload trigger."""
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance
        mock_submit = MagicMock()
        mock_executor_instance.submit.return_value = mock_submit

        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.cloud_name_gcs, self.cloud_bucket, self.cloud_key,
            n_splits=self.n_splits, test_size=self.test_size, upload_cloud_rate=self.upload_rate, **self.params
        )

        # Simulate two trials to trigger upload
        self.mock_trial.number = 0
        optimizer(self.mock_trial)
        self.mock_trial.number = 1
        optimizer(self.mock_trial)

        # Check GCS upload trigger
        self.assertEqual(mock_executor_instance.submit.call_count, 1)
        mock_executor_instance.submit.assert_called_with(optimizer.upload_to_gstorage)
        mock_gcs_client.assert_not_called() # Client itself not called directly in __call__
        mock_s3_client.assert_not_called()

    @patch('gcp_library.optimization_utils.optimizer_ts_allatonce.boto3.client')
    def test_upload_to_s3(self, mock_s3_client):
        """Test the S3 upload method."""
        mock_s3_instance = MagicMock()
        mock_s3_client.return_value = mock_s3_instance
        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.cloud_name_s3, self.cloud_bucket, self.cloud_key,
            **self.params
        )
        optimizer.upload_to_s3()
        mock_s3_client.assert_called_once_with('s3')
        mock_s3_instance.upload_file.assert_called_once_with(self.save_path, self.cloud_bucket, self.cloud_key)

    @patch('gcp_library.optimization_utils.optimizer_ts_allatonce.storage.Client')
    def test_upload_to_gstorage(self, mock_gcs_client):
        """Test the GCS upload method."""
        mock_gcs_instance = MagicMock()
        mock_bucket_instance = MagicMock()
        mock_blob_instance = MagicMock()
        mock_gcs_client.return_value = mock_gcs_instance
        mock_gcs_instance.bucket.return_value = mock_bucket_instance
        mock_bucket_instance.blob.return_value = mock_blob_instance

        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.cloud_name_gcs, self.cloud_bucket, self.cloud_key,
            **self.params
        )
        optimizer.upload_to_gstorage()
        mock_gcs_client.assert_called_once_with()
        mock_gcs_instance.bucket.assert_called_once_with(self.cloud_bucket)
        mock_bucket_instance.blob.assert_called_once_with(self.cloud_key)
        mock_blob_instance.upload_from_filename.assert_called_once_with(self.save_path)

    def test_invalid_cloud_name(self):
        """Test that an invalid cloud name raises an error during upload trigger."""
        optimizer = TimeSeriesOpt(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, "invalid_cloud", self.cloud_bucket, self.cloud_key,
            upload_cloud_rate=1, **self.params
        )
        self.mock_trial.number = 0
        with self.assertRaisesRegex(ValueError, "Unsupported cloud provider: invalid_cloud"):
            optimizer(self.mock_trial)

if __name__ == '__main__':
    unittest.main()
