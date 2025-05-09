# %%
import numpy as np
# from sklearn.model_selection import TimeSeriesSplit
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
            s3_bucket: str, 
            s3_key, 
            **params
        ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.metric = metric
        self.save_path = save_path
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.params = params
        
    def upload_to_s3(self):
        s3 = boto3.client('s3')
        if os.path.exists(self.save_path):
            s3.upload_file(self.save_path, self.s3_bucket, self.s3_key)

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
        
        with open(self.save_path, 'a') as f:
            json.dump({"target": result, "params": params_save}, f)
            f.write('\n')
            
        if (trail.number + 1) % 100 == 0:
            self.upload_to_s3()
        
        return result

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
    s3_key = "optimization_results/results.jsonl" # Replace with your desired S3 key

    optimizer = TimeSeriesOpt(
        X_train_ex, y_train_ex, X_val_ex, y_val_ex,
        ExampleModel, 
        example_metric,
        save_path="results.jsonl",
        s3_bucket=bucket_name,
        s3_key=s3_key,
        **pbounds
    )

    study = optuna.create_study(direction='minimize')
    study.optimize(optimizer, n_trials=500)

    # Print the best parameters and the best value
    print("Best parameters: ", study.best_params)
    print("Best value: ", study.best_value)
# %%
