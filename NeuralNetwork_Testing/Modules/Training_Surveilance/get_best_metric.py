import pandas as pd

class get_best_metric():
    def __init__(self, training_progress: pd.DataFrame): 
        """Get the best RMSE from a DataFrame containing the training progress of a model.
        INDEX = idx
        and
        actual value = best_rmse"""
        self.idx = int(training_progress[training_progress['RMSE'] == training_progress['RMSE'].min()].index[0])
        self.best_rmse = training_progress.loc[self.idx, :].values[0]