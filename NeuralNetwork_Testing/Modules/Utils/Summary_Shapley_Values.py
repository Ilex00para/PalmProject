import os 
import numpy as np

def Summary_Shapley_Values(path='/home/u108-n256/PalmProject/CrossValidation/Model_MNV2_male_alldata'):
    '''Reads the shapley values from the different folds and calculates the mean of the shapley values per kfold sample.
    Sample 1 = Average of sample one of every fold (means each fold had a different model but the same sample 1)
    ...
    Sample n = Average of sample n of every fold (means each fold had a different model but the same sample n)

    args:
    path = path to the folder where the shapley values are stored (should be the CrossValidation/Model_{Modeltype}_{sex}_{data} folder)

    returns:
    None
    saves in the same folder a summary file with the mean of the shapley values per sample
    Shape(5, 7600) where 5 is the number of samples and 7600 is the number of features and their mean value
    '''
    #intilize the array to store the shapley values 
    summary_shaps = np.zeros((4, 25, 7600))
    counter = 0
    #loop over the files in the folder and read the shapley values
    for shapley_file  in os.listdir(path):
        #check if the file is a shapley value file
        if shapley_file.startswith('Fold_3') and shapley_file.endswith('Shapley_values_25.npy'):
            d = np.load(os.path.join(path, shapley_file), allow_pickle=True)
            #loop over the samples and store the shapley values in the summary array
            summary_shaps[counter,:] = d
            counter += 1
            
    #calculate the mean of the shapley values per sample
    summary_shaps = np.mean(summary_shaps, axis=0)
    np.save(os.path.join(path, 'Summary_Shapley_values_25.npy'), summary_shaps)


if __name__ == '__main__':
    for folder in os.listdir('/home/u108-n256/PalmProject/CrossValidation'):
        #print(folder)
        if folder.startswith('Model'):
             Summary_Shapley_Values(os.path.join('/home/u108-n256/PalmProject/CrossValidation', folder))