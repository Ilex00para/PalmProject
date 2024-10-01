from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from pickle import dump, load

def MinMax_Scaling(Phenological_Data: np.array, Meteorological_Data: np.array, global_normalization=True) -> np.array:
    '''
    args:
        Phenological_Data : np.array
            Phenological_Data to be scaled

        id_target_feature : int
            Index of the target feature in the Phenological_Data array

        global_normalization : bool
            if True the maximum and minimum of all datasets are used.
            which is determined in ../NeuralNetwork_Testing/NN_Inputs/Create_NN_Input/normalization_data.py

    '''
    
    
    # Save the original shape of the data
    pheno_shape = Phenological_Data.shape
    meteo_shape = Meteorological_Data.shape

    # Reshape Phenological_Data to 2D array where each row is a sample and each column is a feature
    Phenological_Data = Phenological_Data.reshape(-1, pheno_shape[-1])
    Meteorological_Data = Meteorological_Data.reshape(-1, meteo_shape[-1])
    print(f'Shape of Meteo Data: {Meteorological_Data.shape}')
    
    if global_normalization: 
        print('\nGlobal Normalization is used.\n')
        cwd = os.getcwd()
        print(cwd)
        #load MinMax_scaler 
        with open(os.path.join(cwd,'NeuralNetwork_Testing','NN_Inputs','Scaler_Phenological_global.pkl'),'rb') as f:
            MinMax_Scaler_phenological = load(f)
        with open(os.path.join(cwd,'NeuralNetwork_Testing','NN_Inputs','Scaler_Meteorological_global.pkl'),'rb') as f:
            MinMax_Scaler_meteorological = load(f)

        # Scale the features using MinMaxScaler
        Phenological_Data = MinMax_Scaler_phenological.transform(Phenological_Data).reshape(pheno_shape)
        Meteorological_Data = MinMax_Scaler_meteorological.transform(Meteorological_Data).reshape(meteo_shape)

    else:
        print('\nLocal normalization is used.\n')
        # Initialize MinMax scalers for the features (X) and the target (y)
        MinMax_Scaler_phenological = MinMaxScaler()
        MinMax_Scaler_meteorological = MinMaxScaler()

                # Scale the features using MinMaxScaler
        Phenological_Data = MinMax_Scaler_phenological.fit_transform(Phenological_Data).reshape(pheno_shape)
        Meteorological_Data = MinMax_Scaler_meteorological.fit_transform(Meteorological_Data).reshape(meteo_shape)

    dump(MinMax_Scaler_phenological, open('MinMax_Scaler_phenological.pkl', 'wb'))
    dump(MinMax_Scaler_meteorological, open('MinMax_Scaler_meteorological.pkl', 'wb'))

    # Return the scaled data and the scalers for the target feature and the features
    return Phenological_Data, Meteorological_Data