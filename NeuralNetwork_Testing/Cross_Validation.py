import numpy as np
import numpy as np
from Modules.Main_SCRIPTS.Cross_Validation import cross_validation
from Modules.Datasets import BaseDatasetPalm
from Modules.Architectures import CNN_v0, CNN_MNV2, EfficientNet, ResNet
from Modules.Utils import unpack_masks


if __name__ == '__main__':
    "!!! Not the model just the class since the model is created several times in the cross_validation function!!!"
    model_class = CNN_MNV2

    #dictionary with name of the files which contain the masks
    masks_dict = {
        'All_Data': [None],
        'Excluding flowers male and female' : ['MaleInflo_mask.npy', 'FemaleInflo_mask.npy'],
        'Excluding flowers male and female and bunchload' : ['MaleInflo_mask.npy', 'FemaleInflo_mask.npy', 'BunchLoad_mask.npy'],
        'Excluding bunchload' : ['BunchLoad_mask.npy'],
        'only Meteorological data' : ['Meteorological_mask.npy'],
        'only Phenological data' : ['Phenological_mask.npy'],
        'only BunchLoad data' : ['BunchLoad_only_mask.npy']
    }
    # masks_dict = {
    #     'Meteo plus Bunchload' : ['Meteo_and_Bunchload_mask.npy'],
    #     'Meteo plus flowers' : ['Meteo_and_Flower_mask.npy'],
    # }

    for flower in ['male', 'female']:
        for mask_key in masks_dict:
            #unpack the masks from /NeuralNetwork_Testing/NN_Inputs/NN_Inputs_masks
            mask_list = unpack_masks(masks_dict[mask_key])
            #create the dataset
            dataset = BaseDatasetPalm(flower=flower, train=True, mask=mask_list)
            #folder name to save the models
            folder_name = f'Model_MNV2_{flower}_{mask_key}/'
            
            cross_validation(dataset=dataset,model_class=model_class, kfolds=4, epochs=50, folder_name=folder_name, save_each_model=True)

    with open('./CrossValidation/Mask_dictionary.txt', 'w') as f:
        f.write(f'{masks_dict}\n\n') 