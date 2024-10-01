import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta



def load_data(site:str, dir='/home/u108-n256/PalmProject/NeuralNetwork_Testing'):
    '''
    Load the data from the dataCIGE folder.
    
    Args:
    - site (str): Site identifier to load specific data.

    Returns:
    - Pheno_INPUT (np.array): Phenological data.
    - Meteo_INPUT (np.array): Meteorological data.
    
    !!! Data Features are:
    METEOROLOGICAL DATA (9 features): 
    0'TMin',
    1'TMax',
    2'TAverage',
    3'HRMin',
    4'HRMax',
    5'HRAverage',
    6'WindSpeed',
    7'Rainfall',
    8'Rg'

    PHENOLOGICAL DATA (10 features):
    0'RankOneLeafDate'
    1'AppearedSpatheDate_compl' 
    2'OpenedSpatheDate_compl' 
    3'FloweringDate_compl' 
    4'HarvestDate_compl' 
    5'BunchMass' 
    6'FemaleInflo' 
    7'MaleInflo' 
    8'AbortedInflo' 
    9'BunchLoad
    '''
    def remove_NAN(data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        # Check if the data contains only numeric values
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(f"Data contains non-numeric values: {data.dtype}")

        # Convert NaNs to zeros if any exist
        if np.isnan(data).any():
            data = np.nan_to_num(data)
            
        return data

    folder_path = os.path.join(dir,'dataCIGE')
    
    try: #to load the events on the tree level with numpy
        Pheno_origin = np.load(os.path.join(folder_path, f'data_{site}','Events_tree_{site}_Charge.npy'), allow_pickle=True)
    except pickle.UnpicklingError: #load with pickle
        Pheno_origin = pickle.load(os.path.join(folder_path, f'data_{site}','Events_tree_{site}_Charge.npy'))
    print(f'\nPHENOLOGICAL DATA LOADED\nfrom {folder_path + f"/data_{site}/Events_tree_{site}_Charge.npy"}\nwith Shape {Pheno_origin.shape}\n')

    try:
        Pheno_timeframe = np.array(Pheno_origin[:,:,-1])
        Pheno_INPUT  = np.array(Pheno_origin[:,:,:-1], dtype=np.float32)
    except pd.errors.InvalidIndexError:
        Pheno_timeframe = np.array(Pheno_origin.values[:,:,:,-1])
        Pheno_INPUT  = np.array(Pheno_origin[:,:,:-1].values, dtype=np.float32)

    Pheno_INPUT = remove_NAN(Pheno_INPUT)

    try:
        Meteo_origin = np.load(os.path.join(folder_path, f'data_{site}', 'dfMeteoInfo.pkl'), allow_pickle=True)
    except pickle.UnpicklingError:
        Meteo_origin = pickle.load(os.path.join(folder_path, f'data_{site}', 'dfMeteoInfo.npy'))
    print(f'\nMETEOROLOGICAL DATA LOADED\nfrom {folder_path + f"/data_{site}/dfMeteoInfo.npy"}\nwith Shape {Meteo_origin.shape}\n')

    # Get the column names of the DataFrame
    MeteoVarOfInterest_Init = list(Meteo_origin.columns)

    MeteoVarOfInterest_Init = MeteoVarOfInterest_Init[1:]
    pheno_min_dt = pd.to_datetime(Pheno_timeframe.reshape(-1)).min()
    pheno_max_dt = pd.to_datetime(Pheno_timeframe.reshape(-1)).max()

    
    Meteo_origin = Meteo_origin[Meteo_origin['SiteName'] == site]
    Meteo_origin['ObservationDate'] = pd.to_datetime(Meteo_origin['ObservationDate'])
    Meteo_origin = np.array(Meteo_origin[(Meteo_origin['ObservationDate'] >= pheno_min_dt) & (Meteo_origin['ObservationDate'] <= pheno_max_dt)].values[:,2:], dtype= np.float32)
    Meteo_INPUT = remove_NAN(Meteo_origin)

    """# Plot each variable
    for i in range(Meteo_origin.shape[1]):
        plt.plot(Meteo_origin[1000:2000,i], c='b')
        plt.title(MeteoVarOfInterest_Init[i])
        plt.show()
    """
    if Pheno_origin.shape[1] == Meteo_origin.shape[0]:
        str_out = f'Import of the Data\nThe number of days {Meteo_origin.shape[0]} in the phenology and meteo data is the same'

    if type(Pheno_origin) == type(Meteo_origin):
        str_out = f'The phenology and meteo data have the same data type {type(Pheno_origin)}\n\n'


    if np.isnan(Pheno_INPUT).any() == False and np.isnan(Meteo_INPUT).any() == False:
        str_out = f'The phenology and meteo data do not contain any NaN values\n\n'

    return np.array(Pheno_INPUT), np.array(Meteo_INPUT)