import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

pr_pheno  = np.load('/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/PR/RAW_Phenological_Data.npy')
pr_meteo = np.load('/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/PR/RAW_Meteorological_Data.npy')

smse_pheno  = np.load('/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/SMSE/RAW_Phenological_Data.npy')
smse_meteo = np.load('/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/SMSE/RAW_Meteorological_Data.npy')

print(f'Shape of Meteo Data: {pr_meteo.shape, smse_meteo.shape, smse_meteo[0,:]} ')
print(f'Shape of Pheno Data: {pr_pheno.shape, smse_pheno.shape}')

pheno = np.concatenate((pr_pheno.reshape(-1,pr_pheno.shape[-1]), smse_pheno.reshape(-1,smse_pheno.shape[-1])), axis=0)
meteo = np.concatenate((pr_meteo, smse_meteo), axis=0)

print(pheno.shape, meteo.shape)

MinMax_Scaler_phenological = MinMaxScaler()
MinMax_Scaler_meteorological = MinMaxScaler()

MinMax_Scaler_phenological = MinMax_Scaler_phenological.fit(pheno)
MinMax_Scaler_meteorological = MinMax_Scaler_meteorological.fit(meteo)


with open('/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/Scaler_Phenological.pkl', 'wb') as f:
    pickle.dump(MinMax_Scaler_phenological,f)
with open('/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/Scaler_Meteorological.pkl', 'wb') as f:
    pickle.dump(MinMax_Scaler_meteorological,f)