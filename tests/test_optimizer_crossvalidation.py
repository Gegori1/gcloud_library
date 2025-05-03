import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import json
from gcp_library.optimization_utils.optimizer_crossvalidation import CrossValidationOptimizer
from sklearn.preprocessing import StandardScaler # Import for patching
from sklearn.model_selection import KFold # Import for patching

class TestCrossValidationOptimizer(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(20, 3)
        self.y = np.random.rand(20)
        self.mock_model_cls = MagicMock()
        self.mock_model_instance = MagicMock()
        self.mock_model_cls.return_value = self.mock_model_instance
        self.mock_model_instance.predict.return_value = np.random.rand(4) # Assuming n_folds=5 -> test_size=4

        self.mock_metric = MagicMock(return_value=0.5)
        self.mock_metric.__name__ = "mock_metric"

        self.save_path = "cv_results.jsonl"
        self.cloud_name_s3 = "aws"
        self.cloud_bucket = "test-bucket"
        self.cloud_key = "test/cv_key.jsonl"
        self.param_distributions = {
            'param_float': (0.1, 1.0, "float"),
            'param_int': (1, 10, "int"),
            'param_cat': (['a', 'b'], None, "cat"),
            'param_stat': (5, None, "stationary")
        }
        self.n_folds = 5
        self.upload_rate = 2

        # Mock Optuna trial
        self.mock_trial = MagicMock()
        self.mock_trial.suggest_float.return_value = 0.5
        self.mock_trial.suggest_int.return_value = 5
        self.mock_trial.suggest_categorical.return_value = 'a'
        self.mock_trial.number = 0

        # Mock KFold and StandardScaler
        self.mock_kfold_instance = MagicMock()
        # Generate dummy indices consistent with n_folds and data size
        indices = np.arange(len(self.X))
        fold_indices = []
        fold_size = len(self.X) // self.n_folds
        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_folds - 1 else len(self.X)
            test_idx = indices[test_start:test_end]
            train_idx = np.setdiff1d(indices, test_idx)
            fold_indices.append((train_idx, test_idx))
        self.mock_kfold_instance.split.return_value = fold_indices

        self.mock_scaler_instance = MagicMock()
        self.mock_scaler_instance.fit_transform.side_effect = lambda x: x # Passthrough
        self.mock_scaler_instance.transform.side_effect = lambda x: x # Passthrough

    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.KFold')
    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.StandardScaler')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.boto3.client')
    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.storage.Client')
    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.ThreadPoolExecutor')
    def test_call_with_standardization_s3(self, mock_executor_cls, mock_gcs_client, mock_s3_client, mock_file, mock_scaler_cls, mock_kfold_cls):
        """Test __call__ with standardization enabled and S3 upload."""
        mock_kfold_cls.return_value = self.mock_kfold_instance
        mock_scaler_cls.return_value = self.mock_scaler_instance
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance
        mock_submit = MagicMock()
        mock_executor_instance.submit.return_value = mock_submit

        optimizer = CrossValidationOptimizer(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, self.cloud_name_s3, self.cloud_bucket, self.cloud_key,
            self.param_distributions, n_folds=self.n_folds, standardize=True,
            upload_cloud_rate=self.upload_rate
        )

        # Simulate two trials
        self.mock_trial.number = 0
        result1 = optimizer(self.mock_trial)
        self.mock_trial.number = 1
        result2 = optimizer(self.mock_trial)

        # Assertions
        self.assertEqual(result1, 0.5) # Mean score across folds
        self.assertEqual(result2, 0.5)
        mock_kfold_cls.assert_called_once() # KFold initialized once
        self.assertEqual(self.mock_kfold_instance.split.call_count, 2) # split called per trial
        self.assertEqual(mock_scaler_cls.call_count, 2) # Scaler initialized per trial

        total_fits = self.n_folds * 2 # n_folds per trial
        self.assertEqual(self.mock_scaler_instance.fit_transform.call_count, total_fits)
        self.assertEqual(self.mock_scaler_instance.transform.call_count, total_fits)
        self.assertEqual(self.mock_model_cls.call_count, total_fits)
        self.assertEqual(self.mock_model_instance.fit.call_count, total_fits)
        self.assertEqual(self.mock_model_instance.predict.call_count, total_fits)
        self.assertEqual(self.mock_metric.call_count, total_fits)

        # Check parameter suggestions
        self.mock_trial.suggest_float.assert_called_with('param_float', 0.1, 1.0)
        self.mock_trial.suggest_int.assert_called_with('param_int', 1, 10)
        self.mock_trial.suggest_categorical.assert_called_with('param_cat', ['a', 'b'])

        # Check file writing
        self.assertEqual(mock_file().write.call_count, 2) # Called once per trial
        expected_params_save = {'param_float': 0.5, 'param_int': 5} # Only float/int saved
        expected_json_dump = json.dumps({"target": 0.5, "params": expected_params_save}) + '\n'
        mock_file().write.assert_called_with(expected_json_dump) # Check last call

        # Check S3 upload trigger
        self.assertEqual(mock_executor_instance.submit.call_count, 1) # Called once after trial 1
        mock_executor_instance.submit.assert_called_with(optimizer.upload_to_s3)

    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.KFold')
    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.StandardScaler')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.boto3.client')
    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.storage.Client')
    @patch('gcp_library.optimization_utils.optimizer_crossvalidation.ThreadPoolExecutor')
    def test_call_without_standardization_gcs(self, mock_executor_cls, mock_gcs_client, mock_s3_client, mock_file, mock_scaler_cls, mock_kfold_cls):
        """Test __call__ with standardization disabled and GCS upload."""
        mock_kfold_cls.return_value = self.mock_kfold_instance
        # mock_scaler_cls should NOT be called if standardize=False
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance
        mock_submit = MagicMock()
        mock_executor_instance.submit.return_value = mock_submit

        optimizer = CrossValidationOptimizer(
            self.X, self.y, self.mock_model_cls, self.mock_metric,
            self.save_path, "gcp", self.cloud_bucket, self.cloud_key, # Use GCS name
            self.param_distributions, n_folds=self.n_folds, standardize=False, # Standardize False
            upload_cloud_rate=self.upload_rate
        )

        # Simulate two trials
        self.mock_trial.number = 0
        optimizer(self.mock_trial)
        self.mock_trial.number = 1
        optimizer(self.mock_trial)

        # Assertions
        mock_kfold_cls.assert_called_once()
        self.assertEqual(self.mock_kfold_instance.split.call_count, 2)
        mock_scaler_cls.assert_not_called() # Scaler should not be initialized
        self.assertEqual(self.mock_scaler_instance.fit_transform.call_count, 0)
        self.assertEqual(self.mock_scaler_instance.transform.call_count, 0)

        total_fits = self.n_folds * 2
        self.assertEqual(self.mock_model_cls.call_count, total_fits)
        self.assertEqual(self.mock_model_instance.fit.call_count, total_fits)
        # Check GCS upload trigger
        self.assertEqual(mock_executor_instance.submit.call_count, 1)
        mock_executor_instance.submit.assert_called_with(optimizer.upload_to_gstorage)

if __name__ == '__main__':
    unittest.main()
