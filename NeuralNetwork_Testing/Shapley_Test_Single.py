
import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
from Modules import shapley_test, BaseDatasetPalm, CNN_MNV2, unpack_masks

# uncomment the following code to compute the shapley values for the different models and input data
"""
masks_dict = {
    'All_Data': [[None],['/home/u108-n256/PalmProject/CrossValidation/Model_MNV2_male_All_Data']],
    'only Meteorological data' : [['Meteorological_mask.npy'],['/home/u108-n256/PalmProject/CrossValidation/Model_MNV2_male_only Meteorological data']],
    'only Phenological data' : [['Phenological_mask.npy'],['/home/u108-n256/PalmProject/CrossValidation/Model_MNV2_male_only Phenological data']],
    'only bunchload': [['BunchLoad_only_mask.npy'],['/home/u108-n256/PalmProject/CrossValidation/Model_MNV2_male_only BunchLoad data']],
}

masks_dict = {
    'All_Data': [[None],['/home/u108-n256/PalmProject/CrossValidation/Model_MNV2_male_All_Data']],
}

for mask_key in masks_dict:
    mask_list = unpack_masks(masks_dict[mask_key][0])
    dataset = BaseDatasetPalm(train=False, flower='male', mask=mask_list)

    model_folder_path = masks_dict[mask_key][1][0]
    
    for file in os.listdir(model_folder_path):
        if file.startswith('Fold_') and file.endswith('Model.pt'):
            print('Model set up\n')
            model = CNN_MNV2()
            model.load_state_dict(torch.load(os.path.join(model_folder_path, file)))
            shapley_values = shapley_test(dataset.X, model, random_seed=42, size_background_data=20, nsamples=5) # random seed 42
            np.save(os.path.join(model_folder_path, f'{file.split('_')[0]}_{file.split('_')[1]}_Shapley_values.npy'), shapley_values[0].values)
            np.save(os.path.join(model_folder_path, f'{file.split("_")[0]}_{file.split("_")[1]}_Shapley'), shapley_values[0])
            np.save(os.path.join(model_folder_path, f'{file.split("_")[0]}_{file.split("_")[1]}_Kmeans_samples.npy'), shapley_values[1])
"""



#load the dataset
dataset = BaseDatasetPalm(train=False, flower='male')

#load the model architecture
model = CNN_MNV2()
#load the model weights
model.load_state_dict(torch.load('/home/u108-n256/PalmProject/CrossValidation/OPTIM/Optimized_CNN_best.pt'))

#compute the shapley values and set the number of samples used to calculate the shapley values
shapley_values = shapley_test(dataset.X, model, random_seed=42, size_background_data=20, nsamples=400)

#Save the results
np.save('/home/u108-n256/PalmProject/CrossValidation/OPTIM/Shapley_values.npy', shapley_values[0].values)
df = pd.DataFrame(shapley_values[0].values)
df.to_csv('/home/u108-n256/PalmProject/CrossValidation/OPTIM/Shapley_values.csv')
#np.save(os.path.join(model_folder_path, f'{file.split("_")[0]}_{file.split("_")[1]}_Shapley.npy'), shapley_values[0])

#save also the samples to compare the shapley values with the acual values
np.save('/home/u108-n256/PalmProject/CrossValidation/OPTIM/Kmeans_samples_25.npy', shapley_values[1])
