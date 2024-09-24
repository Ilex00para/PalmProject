import os
import pickle
import torch
import numpy as np
import pandas as pd

from torch import nn

def saving_to_folder(data, directory, folder_name=None, file_name='Saved_file', dtype=None):
    """
    Function to save data to a folder.

    Parameters:
    -----------
    data : various types
        The data to save. Supported types include np.ndarray, torch.Tensor, str, and 'pickle'.
    directory : str
        The directory in which to save the folder.
    folder_name : str, optional
        The name of the folder to save the data. If None, data is saved directly in the directory.
    file_name : str, default 'Saved_file'
        The name of the file to save the data.
    dtype : type, optional
        The data type of the file to save. If None, it is inferred from the type of data provided.

    Returns:
    --------
    None
    """
    
    # Check if the directory exists, create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    folder_path = directory if folder_name is None else os.path.join(directory, folder_name)
    
    # Check if the folder exists, create it if it doesn't
    if folder_name and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Infer data type if not provided
    data_type = type(data) if dtype is None else dtype
    
    counter = 0
    while os.path.exists(os.path.join(folder_path, file_name)):
        counter += 1
        file_name = f"{file_name}_{counter}"

    # Save the data based on its type
    if data_type == np.ndarray:
        np.save(os.path.join(folder_path, file_name + '.npy'), data)
    elif data_type == torch.Tensor:
        torch.save(data, os.path.join(folder_path, file_name + '.pt'))
    elif data_type == nn.Module:
        torch.save(data.state_dict(), os.path.join(folder_path, file_name + '.pt'))
    elif data_type == pd.DataFrame:
        data.to_csv(os.path.join(folder_path, file_name + '.csv'), index=False)
    elif data_type == str:
        file_path = os.path.join(folder_path, file_name + '.txt')
        if os.path.exists(file_path):
            print("File already exists. Appending to the file.")
            with open(file_path, 'a') as f:
                f.write(data)
        else:
            with open(file_path, 'w') as f:
                f.write(data)
    elif data_type == 'pickle':
        with open(os.path.join(folder_path, file_name + '.pkl'), 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    print(f"Data saved to: {os.path.join(folder_path, file_name)}")
