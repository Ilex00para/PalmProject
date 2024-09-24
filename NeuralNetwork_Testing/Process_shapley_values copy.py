# %%
import shap
import shap.maskers
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

import NN_Architectures as arch
from Functions.DataSet import PhenologicalDatasetTransposedMA
from sklearn.cluster import KMeans
# %%
def get_no_phenological_features(shap_values):
    try:
        shap_values = shap_values.reshape(-1, 40, 190)
        n_pheno_features = 10
    except:
        try:
            shap_values = shap_values.reshape(-1, 40, 188)
            n_pheno_features = 8
        except:
            shap_values = shap_values.reshape(-1, 40, 187)
            n_pheno_features = 7
    return shap_values, n_pheno_features

# %%
def get_all_shapley_values_per_feature(shap_values, n_pheno_features):
    all_features_shaps = []

    #reshape shap values to be able to iterate over samples, 20 day periods and features

    #initialize list of list of shap values for each feature
    for i in range(9+n_pheno_features):
        all_features_shaps.append([])

    for sample in range(shap_values.shape[0]):

        for tweenty_day_period in range(shap_values.shape[1]):

            for pheno_features in range(0,n_pheno_features):
                all_features_shaps[pheno_features].append(shap_values[sample, tweenty_day_period, pheno_features])
            
            for days in range(20):

                for weather_features in range(0,9):
                    # since days starts with 0 and weather_features with 10, we need to multiply days by 10 to get the correct index
                    all_features_shaps[weather_features+n_pheno_features].append(shap_values[sample, tweenty_day_period, n_pheno_features + weather_features + days*9])
    return all_features_shaps

# %%
def time_wise_shaps(shap_values):
    '''Takes in shapley values and returns the average and sum of the shap values for each 20_day_period
    args: 
    shap_values: numpy array of shapley values with shape (samples, 35, 188)
    return:
    avg_abs_time_shaps: numpy array of average of absolute shap values for each 20_day_period
    avg_time_shaps: numpy array of average of shap values for each 20_day_period    
    sum_abs_time_shaps: numpy array of sum of absolute shap values for each 20_day_period   
    sum_time_shaps: numpy array of sum of shap values for each 20_day_period
    '''
    

    #create list with 35 empty lists
    time_shaps_avg = []
    time_shaps_sum = []
    time_abs_shaps_avg = []
    time_abs_shaps_sum = []
    for i in range(shap_values.shape[1]):
        time_shaps_avg.append([])
        time_shaps_sum.append([])
        time_abs_shaps_avg.append([])
        time_abs_shaps_sum.append([])

    #loop samples
    for sample in range(shap_values.shape[0]):
        #loop 35 20_day_periods
        for tweenty_day_period in range(shap_values.shape[1]):
            #append the mean of the shap_values of the 188 features 
            time_shaps_avg[tweenty_day_period].append(np.mean(shap_values[sample, tweenty_day_period,:]))
            time_shaps_sum[tweenty_day_period].append(np.sum(shap_values[sample, tweenty_day_period,:]))

            time_abs_shaps_avg[tweenty_day_period].append(np.mean(np.abs(shap_values[sample, tweenty_day_period,:])))
            time_abs_shaps_sum[tweenty_day_period].append(np.sum(np.abs(shap_values[sample, tweenty_day_period,:])))
    
    avg_abs_time_shaps = np.zeros(len(time_shaps_avg))
    avg_time_shaps = np.zeros(len(time_shaps_avg))
    sum_abs_time_shaps = np.zeros(len(time_shaps_sum))
    sum_time_shaps = np.zeros(len(time_shaps_sum))

    # calculate the sum and mean of the shap values for each 20_day_period
    for i, data in enumerate(zip(time_abs_shaps_sum, time_shaps_sum)):
        sum_abs_time_shaps[i] = np.sum(np.array(data[0]).__abs__())
        sum_time_shaps[i] = np.mean(np.array(data[1]))



    return (avg_abs_time_shaps, avg_time_shaps, sum_abs_time_shaps, sum_time_shaps)

# %%
def feature_wise_shaps(all_features_shaps, n_pheno_features):
    '''Takes in all feature shapley values and returns several sorted versions of it

    args:
    all_features_shaps: list of lists of shap values for each feature

    return:
    all_features_shaps: list of numpy arrays of shap values for each feature
    
    abs_sum_feature_shaps: sum of absolute shap values for each feature (total strength of the feature) 
    sum_feature_shaps: sum of shap values for each feature (total contribution of the feature, plus or minus tendency)

    abs_avg_feature_shaps: average of absolute shap values for each feature (average strength of a feature per data point)
    avg_feature_shaps: average of shap values for each feature (average contribution of a feature per data point)
    '''
    abs_sum_feature_shaps = np.zeros(9+n_pheno_features)
    sum_feature_shaps = np.zeros(9+n_pheno_features)
    abs_avg_feature_shaps = np.zeros(9+n_pheno_features)
    avg_feature_shaps = np.zeros(9+n_pheno_features)

    for i, data in enumerate(all_features_shaps):
        
        all_features_shaps[i] = np.array(data)

        abs_sum_feature_shaps[i] = np.sum(np.array(data).__abs__())
        sum_feature_shaps[i] = np.sum(np.array(data))

        abs_avg_feature_shaps[i] = np.mean(np.array(data).__abs__())
        avg_feature_shaps[i] = np.mean(np.array(data))

    return (all_features_shaps, abs_sum_feature_shaps, sum_feature_shaps, abs_avg_feature_shaps, avg_feature_shaps)

# %%
def load_shapley_vals(path):
    ''' Loads shapley values from a file and returns several sorted versions of it

    args: 
    path: path to the file containing the shapley values

    Sorted versions include:
    output: tuple of two tuples
    output[0]:tuple of five numpy arrays, feature wise shapley values
        output[0][0]: all_features_shaps: list of numpy arrays of shap values for each feature (time series for each feature)
    
        output[0][1]: abs_sum_feature_shaps: sum of absolute shap values for each feature (total strength of the feature, data point for each feature) 
        output[0][2]: sum_feature_shaps: sum of shap values for each feature (total contribution of the feature, plus or minus tendency)

        output[0][3]: abs_avg_feature_shaps: average of absolute shap values for each feature (average strength of a feature per data point, data point for each feature)
        output[0][4]:avg_feature_shaps: average of shap values for each feature (average contribution of a feature per data point)
        
    output[1]: tuple of 4 numpy arrays, time wise shapley values    
        output[1][0]: avg_abs_time_shaps: numpy array of average of absolute shap values for each 20_day_period (data point for each 20_day_period)
        output[1][1]: avg_time_shaps: numpy array of average of shap values for each 20_day_period (data point for each 20_day_period)
        output[1][2]: sum_abs_time_shaps: numpy array of sum of absolute shap values for each 20_day_period (data point for each 20_day_period) 
        output[1][3]: sum_time_shaps: numpy array of sum of shap values for each 20_day_period (data point for each 20_day_period)
    output[2]: shapley_values: numpy array of shapley values
    output[3]: n_pheno_features: number of phenological features in the dataset
    output[4]: samples: number of samples in the dataset
        '''
    shapley_values = np.load(path, allow_pickle=True)
    samples = shapley_values.shape[0]

    shap_values, n_pheno_features = get_no_phenological_features(shapley_values)
    all_features_shaps = get_all_shapley_values_per_feature(shap_values, n_pheno_features)
    feature_wise = feature_wise_shaps(all_features_shaps, n_pheno_features)
    time_wise = time_wise_shaps(shap_values)
    
    out = (feature_wise, time_wise, shap_values,n_pheno_features,samples)
    out_path = path
    for f in range(2):
        out_path = os.path.dirname(out_path)
    out_path = os.path.join(out_path, 'processed_shapley_values', path.split('/')[-1].split('.')[0] + '_processed.pkl')
    pickle.dump(out, open(out_path, 'wb'))
    print(path.split('/')[-1].split('.')[0], out[0][0].__len__(), out[1][0].__len__())
    return out

for file in os.listdir('/home/u108-n256/PalmProject/NeuralNetwork_Testing/Saved_Objects/Shapley-Values/raw_shapley_values'):
    if file.endswith('.npy'):
            load_shapley_vals(os.path.join('/home/u108-n256/PalmProject/NeuralNetwork_Testing/Saved_Objects/Shapley-Values/raw_shapley_values/', file))