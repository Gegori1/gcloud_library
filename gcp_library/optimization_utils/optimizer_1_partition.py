# %%
import numpy as np
from google.cloud import storage
from sklearn.model_selection import train_test_split
import boto3
import optuna
import json
import os
# from .s3_utils import upload_to_s3

class TimeSeriesOpt:
    """
    Class to define the function used for the Bayesian Optimization
    for time series data, using pre-split train and validation sets.
    """
    def __init__(
            self, X_train, y_train, X_val, y_val,
            model, 
            metric, 
            save_path: str,
            cloud_name: str,
            cloud_bucket: str, 
            cloud_key: str,
            param_config: dict,
            direction: str="minimize",
            n_jobs: int=1,
            upload_to_cloud: bool=False,
            upload_cloud_rate: int=100
        ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.metric = metric
        self.save_path = save_path
        self.cloud_name = cloud_name
        self.cloud_bucket = cloud_bucket
        self.cloud_key = cloud_key
        self.direction = direction
        self.n_jobs = n_jobs
        self.upload_to_cloud = upload_to_cloud
        self.params = param_config
        self.upload_cloud_rate = upload_cloud_rate
        
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

    def __call__(self, trail):
        
        params = {}
        for param_name, (low, high, ptype) in self.params.items():
            if ptype == "float":
                params[param_name] = trail.suggest_float(param_name, low, high)
            elif ptype == "int":
                params[param_name] = trail.suggest_int(param_name, low, high)
            elif ptype == "cat":
                params[param_name] = trail.suggest_categorical(param_name, low)
            elif ptype == "stationary":
                params[param_name] = low
            else:
                raise ValueError(f"Unsupported parameter type: {ptype}")
        
        model = self.model(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val).flatten()
        
        result = self.metric(self.y_val, y_pred)
        
        params_save = {
            param_name: param_value
            for param_name, param_value 
            in params.items()
            if isinstance(param_value, (int, float, str))
        }
        
        self.save_file(params_save, result)
            
        if (self.upload_to_cloud) and ((trail.number + 1) % self.upload_cloud_rate == 0):
            self.upload_to_s3()
        elif (not self.upload_to_cloud) and ((trail.number + 1) % self.upload_cloud_rate == 0):
            print("Canceling upload. upload_to_cloud set to False")
        
        return result
    
    def optimize(self, n_trials):
        """
        Run the Optuna optimization study.
        """
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=60)
        study = optuna.create_study(direction=self.direction, sampler=sampler)
        study.optimize(self, n_trials=n_trials, n_jobs=self.n_jobs)
        
        print("Best parameters: ", study.best_params)
        print("Best value: ", study.best_value)
        return study

# %% Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    X_full = np.random.randn(100, 2)
    y_full = np.random.randint(0, 2, 100)

    # Pre-split the data for the example
    X_train_ex, X_val_ex, y_train_ex, y_val_ex = train_test_split(
        X_full, y_full, test_size=0.3, shuffle=False
    )

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

    # Example: Define bucket_name and s3_key if you intend to run the example with S3 upload
    bucket_name = "your-s3-bucket-name" # Replace with your actual bucket name
    cloud_key = "optimization_results/results.jsonl" # Replace with your desired cloud key

    optimizer = TimeSeriesOpt(
        X_train_ex, y_train_ex, X_val_ex, y_val_ex,
        ExampleModel, 
        example_metric,
        save_path="results.jsonl",
        cloud_name="default",
        cloud_bucket=bucket_name,
        cloud_key=cloud_key,
        n_jobs=-1,
        upload_to_cloud=True,
        param_config=pbounds,
    )
    
    optimizer.optimize(n_trials=500)
# %%
