import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Union, List, Sequence, Tuple

class BaseDatasetPalm(Dataset):
    def __init__(self, 
                 flower: str, 
                 site: Union[List, Tuple] = 'SMSE',
                 path: str = '/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs', 
                 train: bool = True, 
                 random_seed: int = 42, 
                 time_windows: list = [0, 39, 49], 
                 mask: Union[np.array, List[np.array]] = None):
        """
        Initialize the BaseDatasetPalm object.

        Args:
            flower (str): 'female', 'male', or 'aborted'. Flower type which should be predicted (sets the target variable y).
            path (str): Path to the NN_Inputs folder which contains the NN_Inputs_raw.npy file. Defaults to a specific path.
            train (bool): If True, training data will be loaded. If False, test data will be loaded. Defaults to True.
            random_seed (int): Random seed for reproducibility. Defaults to 42.
            time_windows (tuple): Time windows for the data can be adpated in the following way:
                                  tuple of time windows index [0]: start date for X data (default 1 20-day period, index 0 of the data array)
                                  tuple of time windows index [1]: end date for X data (default 40 20-day period, index 39 of the data array)
                                  tuple of time windows index [2]: start date for y data (default 50 20-day period, index 49 of the data array)
            mask (np.array): Mask to exclude certain features from the data. Should be a True/False array (False excludes feature).

        Raises:
            ValueError: If flower is None.
            AssertionError: If time windows are invalid or do not match data shape.
        
        Returns:
            self - Instance of the BaseDatasetPalm object.
        """
        self.flower = flower if flower is not None else ValueError('Please provide a flower')
        self.flower_dict = {'female': 6, 'male': 7, 'aborted': 8}

        assert isinstance(site, (list, tuple)), 'Site should be a list or tuple'
        self.path = [os.path.join(path,s) for s in site]
        self.train = train
        self.random_seed = random_seed
        self.time_windows = time_windows
        self.mask = mask
        self.X = None 
        self.y = None
        self.X_n_y()

    def X_n_y(self):
        """Updates the self.X and self.y data depending on the number of sites.
        args:
            None
        returns:
            None
        """
        #loop over the different sites
        for path in self.path:

            print(f'Retrieving data from {path}')
            # Load the data of a site
            self.data = np.load(os.path.join(path, 'NN_Inputs_raw.npy'), allow_pickle=True)

            # Split the data into training and test sets
            training_data, test_data = train_test_split(self.data, test_size=0.2, random_state=self.random_seed, shuffle=True)
            data = training_data if self.train else test_data

            y = self.define_y(data)
            self.concatenate_X(data, y)

    def define_y(self, data):
        '''
        Select the y data based on the flower type and time window.
        args:
            data (np.array): Data to select the y data from.
        returns:
            y (np.array): y data for the given flower type and time window.
        '''
        if self.time_windows[2] >= 50:
            raise NotImplementedError('The current time window is 50 days')
        
        if self.time_windows[2] == 49:
            y = data[:, :, self.time_windows[2], self.flower_dict[self.flower]].reshape(-1, 1)
        else:
            # If the time window is not the last time window, select all the data after the time window
            y = data[:, :, self.time_windows[2]:, self.flower_dict[self.flower]].reshape(-1, self.data.shape[2] - self.time_windows[2],1)
        return y

    def concatenate_X(self, data, y):
        '''Concatenate the X and y data and update the self.X and self.y data.'''
        # Concatenate the X and the y data
        if self.X is None:
            self.X = self.masking_X(data).reshape(-1, (self.time_windows[1]+1) - self.time_windows[0], self.data.shape[3])
            self.y = y
        else:
            # !! Is already cutted into time frame smaples in the masking_X function!!
            self.X = np.concatenate((self.X, self.masking_X(data).reshape(-1, (self.time_windows[1]+1) - self.time_windows[0], self.data.shape[3])))
            self.y = np.concatenate((self.y, y))
    

    def masking_X(self, data):
        '''Apply a mask to the data to exclude certain features.
        args:  
            data (np.array): Data to apply the mask to.
        returns:
            data (np.array): Masked data.
        '''

        if self.mask is not None:
            if type(self.mask) == np.array:
                self.mask = [self.mask]
            for m in self.mask:
                # Verify that the mask shape matches the data shape
                assert m.shape[0] == self.data.shape[2], f'Mask does not match data shape. Mask shape: {m.shape[0]} Data shape: {self.data.shape[2]}'
                assert m.shape[1] == self.data.shape[3], f'Mask does not match data shape. Mask shape: {m.shape[1]} Data shape: {self.data.shape[3]}'
                #adds additional dimensions to the mask for broadcasting
                apply_mask = np.expand_dims(np.expand_dims(m,axis=0),axis=0)
                data = data * apply_mask
        data = data[:, :, self.time_windows[0]:self.time_windows[1]+1, :]
        return data


    def __len__(self):
        """Return the total number of samples in the dataset."""
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f'Length mismatch: X has {self.X.shape[0]} samples, but y has {self.y.shape[0]} samples')

        return self.y.shape[0]


    def __getitem__(self, idx):
        '''Return a sample of the X and y data for a given index.
        args:
            idx (int): Index of the sample to return.
        returns:
            X (torch.tensor): X data for the sample with shape [40, 190] = [20-day-periods, features].
            y (torch.tensor): y data for the sample with shape [1] = [number].
        '''
        return (
            torch.tensor(self.X[idx], dtype=torch.float32), 
            torch.tensor(self.y[idx], dtype=torch.float32)
            )
    
# for testing purposes
if __name__ == '__main__':
    dataset = BaseDatasetPalm(flower='male', site=['SMSE','PR'], train=False)
    print(dataset.__getitem__(100)[0])
    print(dataset.__getitem__(100)[1])
    print(dataset.__len__())