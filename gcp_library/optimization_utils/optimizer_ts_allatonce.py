# %%
import boto3
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import optuna
import json
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor

class TimeSeriesOpt:
    """
    Class to define the function used for the Bayesian Optimization
    for time series data.
    """
    def __init__(self, X, y, model, metric, save_path: str, cloud_name, cloud_bucket, cloud_key, n_splits=5, test_size=None, gap=0, upload_cloud_rate=100, **params):
        self.X = X
        self.y = y
        self.model = model
        self.metric = metric
        self.save_path = save_path
        self.cloud_name = cloud_name
        self.cloud_bucket = cloud_bucket
        self.cloud_key = cloud_key
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.params = params
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        self.upload_cloud_rate = upload_cloud_rate
        self.executor = ThreadPoolExecutor(max_workers=4)  
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
        params = {}
        for param_name, (low, high, ptype) in self.params.items():
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
        for train_index, test_index in self.tscv.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            model = self.model(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).flatten()
            
            score = self.metric(y_test, y_pred)
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
            if isinstance(param_value, (int, float, str))
        }
        # save parameters
        self.save_file(params_save, score)
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
    
    X = np.random.randn(12, 2)
    y = np.random.randint(0, 2, 12)

    def example_metric(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    # Define the model 
    class ExampleModel:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.random.randn(len(X))
        
    # Define the parameter bounds
    pbounds = {
        'param1': (0.1, 1.0, "float"),
        'param2': (-1, 1, "int"),
        'param3': (["rbf", "sinc", "linear"], None, "cat"),
        'param4': (object, None, "stationary")
    }

    optimizer = TimeSeriesOpt(
        X, y, 
        ExampleModel, 
        example_metric,
        n_splits=2, 
        test_size=None, 
        gap=3, 
        minimize=True, **pbounds
    )

    study = optuna.create_study(direction='minimize')


    # Optimize the study
    study.optimize(optimizer, n_trials=1)

    # Print the best parameters and the best value
    print("Best parameters: ", study.best_params)
    print("Best value: ", study.best_value)
# %%
