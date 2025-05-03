# %%
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import optuna
import json
import os
import boto3

class TimeSeriesOpt:
    """
    Class to define the function used for the Bayesian Optimization
    for time series data with rolling splits and hyperparameter saving.
    """
    def __init__(
            self, X, y, 
            model, metric, 
            save_path: str,
            s3_bucket: str, 
            s3_key: str,
            n_splits=5, 
            test_size=None, 
            gap=0,
            minimize=True,
            upload_s3_rate=100,
            **params
        ):
        self.X = X
        self.y = y
        self.model = model
        self.metric = metric
        self.save_path = save_path
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.params = params
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.minimize = minimize
        self.upload_s3_rate = upload_s3_rate

    def upload_to_s3(self):
        s3 = boto3.client('s3')
        base, ext = os.path.splitext(self.save_path)
        save_path = f"{base}_{self.split_number}{ext}"
        base, ext = os.path.splitext(self.s3_key)
        s3_key = f"{base}_split_{self.split_number}{ext}"
        # print(save_path, flush=True)
        # print(s3_key, flush=True)
        s3.upload_file(save_path, self.s3_bucket, s3_key)

    def get_params(self, trail):
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
            
        return params
            
    def save_file(self, params_save, target):
        base, ext = os.path.splitext(self.save_path)
        save_path = f"{base}_{self.split_number}{ext}"
        with open(save_path, 'a') as f:
            json.dump({"target": target, "params": params_save}, f)
            f.write('\n')
        
    def objective(self, trial, X_train, y_train, X_test, y_test):
        params = self.get_params(trial)
        model = self.model(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).flatten()
        score = self.metric(y_test, y_pred)
        
        params_save = {
            param_name: param_value
            for param_name, param_value 
            in params.items()
            if isinstance(param_value, (int, float, str))
        }
        
        # save parameters
        self.save_file(params_save, score)
        if (trial.number + 1) % self.upload_s3_rate == 0:
            self.upload_to_s3()
        
        return score

    def optimize_splits(self, n_trials=10):
        direction = "minimize" if self.minimize else "maximize"
        for split_number, (train_index, test_index) in enumerate(self.tscv.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.split_number = split_number

            study = optuna.create_study(direction=direction)
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials, n_jobs=-1)


# %% Example usage
if __name__ == "__main__":
    
    X = np.random.randn(20, 2)
    y = np.random.randint(0, 2, 20)

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
    
    bucket_name = "your-bucket-name"
    s3_key = "Optimization/optuna_ts_rolling.jsonl"

    optimizer = TimeSeriesOpt(
        X, y, 
        ExampleModel, 
        example_metric,
        save_path="rolling_results.jsonl",
        s3_bucket=bucket_name,
        s3_key=s3_key,
        n_splits=3, 
        test_size=3, 
        gap=0, 
        **pbounds
    )

    # Optimize the study
    optimizer.optimize_splits(n_trials=5)
# %%
