import numpy as np
import os
from Create_NN_Input import load_data
from Create_NN_Input import process_phenological_data
from Create_NN_Input import MinMax_Scaling
from Create_NN_Input import merge_samples_meteo_pheno
from sklearn.model_selection import train_test_split

"""This script is used to create the input data for the neural network model. 
Reads data from directory.
default: '/home/u108-n256/PalmProject/NeuralNetwork_Testing/'

RAW DATA are saved in site folder.

Phenological_Data = The phenological data is saved as a numpy array.
Meteorological_Data = The meteorological data is saved as a numpy array.
NN_inputs = The nn input data is saved as a numpy array.

"""

site = ['PR', 'SMSE']

def main(site):
    current_wkdir = os.getcwd()
    print(current_wkdir)
    #This loads the data from the directory /home/u108-n256/PalmProject/NeuralNetwork_Testing and brings them in a 
    #constant format of numpy.arrays
    Phenological_Data, Meteorological_Data = load_data(dir=os.path.join(current_wkdir,'NeuralNetwork_Testing'),site=site)

    #Saving the raw data /home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/{site}/RAW_Phenological_Data.npy
    np.save(os.path.join(current_wkdir,'NeuralNetwork_Testing','NN_Inputs',site,'RAW_Phenological_Data'), Phenological_Data)
    np.save(os.path.join(current_wkdir,'NeuralNetwork_Testing','NN_Inputs',site,'RAW_Meteorological_Data'), Meteorological_Data)


    """_________From here on the data sets need to be used separately as the normalizations is done separately.____________"""

    #Compressing the phenological data from days to 20-day-periods
    Phenological_Data = process_phenological_data(Phenological_Data, X=20, apply_moving_average=True)

    #Normalizing the data (orientattion on the minx-max of the site/location)
    Phenological_Data, Meteorological_Data = MinMax_Scaling(Phenological_Data, Meteorological_Data, global_normalization=True)

    #Merging the phenological and meteorological data to one dataset
    NN_inputs = merge_samples_meteo_pheno(Phenological_Data, Meteorological_Data, X=20)

    #Saving the complete dataset which is can be used for the neural network
    np.save(os.path.join(current_wkdir,'NeuralNetwork_Testing','NN_Inputs',site,'NN_Inputs_raw'), NN_inputs)

    #Splitting the data into training and testing data
    NN_IN_Train, NN_IN_Test = train_test_split(NN_inputs, test_size=0.2, random_state=42, shuffle=False)

    np.save(os.path.join(current_wkdir,'NeuralNetwork_Testing','NN_Inputs',site,'NN_Inputs_Train'), NN_IN_Train)
    np.save(os.path.join(current_wkdir,'NeuralNetwork_Testing','NN_Inputs',site,'NN_Inputs_Test'), NN_IN_Test)

if __name__ == '__main__':
    for s in site:
        assert isinstance(s, str), 'Sitename needs to be a string'
        main(s)