# %%
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
from concurrent.futures import ThreadPoolExecutor
import json
import boto3
from google.cloud import storage

def mape(y_true, y_pred):
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters:
    - y_true: array-like of actual values
    - y_pred: array-like of predicted values

    Returns:
    - SMAPE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

class CrossValidationOptimizer:
    """
    Class to optimize hyperparameters using cross-validation, to be used as objective function in Optuna.
    """
    def __init__(self, X, y, model, metric, save_path: str, cloud_name, cloud_bucket, cloud_key, param_distributions, n_folds=5, standardize=True, random_state=42, upload_cloud_rate=100):
        """
        Initialize the CrossValidationOptimizer.

        Parameters:
        - X: array-like of features
        - y: array-like of target values
        - model: sklearn model to optimize
        - save_path: path to save the results
        - cloud_name: name of the cloud provider
        - cloud_bucket: name of the cloud bucket
        - cloud_key: key to save the file in the cloud bucket
        - param_distributions: dictionary of parameters to optimize. Optuna will suggest values for these.
          The format should be: {'param_name': (low, high, type)}, where type is 'float', 'int', or 'categorical'.
        - n_folds: number of folds for cross-validation
        - standardize: whether to standardize the data
        - random_state: random state for reproducibility
        - upload_cloud_rate: rate to upload the file to the cloud
        """
        self.X = X
        self.y = y
        self.model = model
        self.metric = metric
        self.save_path = save_path
        self.cloud_name = cloud_name
        self.cloud_bucket = cloud_bucket
        self.cloud_key = cloud_key
        self.param_distributions = param_distributions
        self.n_folds = n_folds
        self.standardize = standardize
        self.random_state = random_state
        self.kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.upload_cloud_rate = upload_cloud_rate
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers as needed
        self.upload_futures = []

    def save_file(self, params_save, result):
        with open(self.save_path, 'a') as f:
            json.dump({"target": result, "params": params_save}, f)
            f.write('\n')

    def upload_to_s3(self):
        s3 = boto3.client('s3')
        s3.upload_file(self.save_path, self.cloud_bucket, self.cloud_key)
        
    def upload_to_gstorage(self):
        client = storage.Client()
        bucket = client.bucket(self.cloud_bucket)
        blob = bucket.blob(self.cloud_key)
        blob.upload_from_filename(self.save_path)

    def __call__(self, trial):
        """
        Objective function for Optuna.

        Parameters:
        - trial: Optuna trial object

        Returns:
        - score: SMAPE score for the given hyperparameters
        """
        scaler = StandardScaler()
        params = {}
        for param_name, (low, high, ptype) in self.param_distributions.items():
            if ptype == "float":
                params[param_name] = trial.suggest_float(param_name, low, high)
            elif ptype == "int":
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif ptype == "cat":
                params[param_name] = trial.suggest_categorical(param_name, low)
            elif ptype == "stationary":
                params[param_name] = low
            else:
                raise ValueError(f"Unsupported parameter type: {ptype}")

        scores = []
        for train_index, val_index in self.kf.split(self.X, self.y):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            # Standardize the data
            if self.standardize:
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            # Train and evaluate the model
            model = self.model(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val).flatten()

            score = self.metric(y_val, y_pred)
            try:
                if self.metric.__name__ == "mean_absolute_percentage_error":
                    score *= 100
                
            except AttributeError:
                self.metric.__name__ = "metric_name"
            
            scores.append(score)

        result = np.mean(scores)
        
        params_save = {
            param_name: param_value
            for param_name, param_value 
            in params.items()
            if isinstance(param_value, (int, float))
        }
        # save parameters
        self.save_file(params_save, result)
        print(f"trail number {trial.number}", flush=True)
        if (trial.number + 1) % self.upload_cloud_rate == 0:
            if self.cloud_name.lower() in ["amazon", "aws"]:
                future = self.executor.submit(self.upload_to_s3)
                self.upload_futures.append(future)
                print("File upload to s3 submitted", flush=True)
            elif self.cloud_name.lower() in ["google", "gcp"]:
                future = self.executor.submit(self.upload_to_gstorage)
                self.upload_futures.append(future)
                print("File upload to gstorage submitted", flush=True)
            else:
                raise ValueError(f"Unsupported cloud provider: {self.cloud_name}")
        
        self._wait_for_uploads()
        return result

    def _wait_for_uploads(self):
        """Waits for all pending uploads to complete."""
        for future in self.upload_futures:
            future.result()  # Wait for the upload to finish
        self.upload_futures = []  # Clear the list of futures

# %% Example usage
if __name__ == "__main__":
    import optuna
    from sklearn.svm import SVR
    import numpy as np
    
    # Generate some dummy data
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    # Define the parameter distributions for Optuna
    param_distributions = {
        'C': (1e-5, 1e2, 'float'),
        'gamma': (1e-5, 1e1, 'float'),
        'kernel': (['linear', 'rbf'], None, 'cat')
    }
    
    # Define the saving parameters
    save_path = "optimization_results.jsonl"
    cloud_name = "gcp"  # or "aws"
    cloud_bucket = "your-bucket-name"  
    cloud_key = "optimization_results.jsonl"

    # Create the CrossValidationOptimizer instance
    optimizer = CrossValidationOptimizer(
        X, y,
        SVR,
        mean_absolute_percentage_error,
        save_path,
        cloud_name,
        cloud_bucket,
        cloud_key,
        param_distributions,
        n_folds=3,
        standardize=True,
        random_state=42
    )

    # Create an Optuna study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(optimizer, n_trials=10)

    # Print the results
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

# %%
