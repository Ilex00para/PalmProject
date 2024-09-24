import numpy as np
import torch
from torch.utils.data import Dataset

class CrossValidationDataset(Dataset):
    '''Just normal dataset class taking X and y used in the cross validation.
        X and y are numpy arrays created from the folds in the cross validation.'''
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]