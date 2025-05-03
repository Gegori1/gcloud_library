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
    for time series data.
    """
    def __init__(
            self, X, y, 
            model, 
            metric, 
            save_path: str,
            s3_bucket: str, 
            s3_key, 
            test_size=None,
            **params
        ):
        self.X = X
        self.y = y
        self.model = model
        self.metric = metric
        self.save_path = save_path
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.test_size = test_size
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
            
        
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, 
            test_size=self.test_size,
            shuffle=False
        )  
        
        model = self.model(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val).flatten()
        
        result = self.metric(y_val, y_pred)
        
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
        test_size=0.3,
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
