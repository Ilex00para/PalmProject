import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import sklearn 
import random
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Sequence, Tuple
import pickle
import scipy.stats

# %%
meteo = np.load('/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/PR/RAW_Meteorological_Data.npy')
pheno = np.load('/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/PR/RAW_Phenological_Data.npy')


# %%
# Define the number of points in the kernel and standard deviation



kernel_size = 17  # Number of points in the kernel (should be odd to have a center point)
std_dev = 2  # Standard deviation, controls the spread of the Gaussian curve

# Generate a linspace array centered at 0 (kernel will be symmetrical around 0)
x_values = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)

# Compute the Gaussian kernel using the normal distribution's PDF
gaussian_kernel = scipy.stats.norm.pdf(x_values, loc=0, scale=std_dev)

# Normalize the kernel to have the highest value as 1 (optional, but useful)
gaussian_kernel /= gaussian_kernel.max()

# %%
columns = ['RankOneLeafDate','AppearedSpatheDate_compl','OpenedSpatheDate_compl','FloweringDate_compl','HarvestDate_compl','BunchMass','FemaleInflo','MaleInflo','AbortedInflo','BunchLoad']
pheno = [pd.DataFrame(pheno[i], columns=columns) for i in range(len(pheno))]

# for p in pheno:
#     p['FemaleInflo_r'] = p['FemaleInflo'].rolling(window=17,center=True).apply(lambda x: np.sum(x*gaussian_kernel), raw=True).fillna(0)
#     p['MaleInflo_r'] = p['MaleInflo'].rolling(window=17,center=True).apply(lambda x: np.sum(x*gaussian_kernel), raw=True).fillna(0)

for p in pheno:
    p['FemaleInflo_r'] = p['FemaleInflo'].rolling(window=40,center=True).apply(lambda x: np.mean(x) if np.any(x > 0) else 0, raw=True).fillna(0)
    p['MaleInflo_r'] = p['MaleInflo'].rolling(window=40,center=True).apply(lambda x: np.mean(x) if np.any(x > 0) else 0, raw=True).fillna(0)


columns=['TMin', 'TMax', 'TAverage', 'HRMin', 'HRMax', 'HRAverage', 'WindSpeed', 'Rainfall','Rg']
meteo = pd.DataFrame(meteo, columns=columns)

class RandomForestDataset():
    def __init__(self,meteo_data: pd.DataFrame,pheno_data: Sequence[pd.DataFrame]) -> None:
        """
        Function which initializes the dataset for the Random Forest
        Args:
        meteo_data: pd.DataFrame, the meteorological data
        pheno_data: Sequence[pd.DataFrame], the phenological data
        
        Returns:
        self
        """
        self.data = [pd.concat([meteo_data,pheno], axis=1).dropna(axis=0) for pheno in pheno_data]

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self,tree:int) -> Sequence[pd.DataFrame]:
        return self.data[tree]
    
    def get_dataset(self,start_date, end_date, prediction_start, prediction_end, random_sample:int=None) -> pd.DataFrame:
        """
        Function which retrieves the dataset from the given start and end date for the data X
        and the prediction_start and prediction_end for the prediction y.
        The dates give the relative distance and keep it for every sample.
        Random samples can be taken from the dataset.

        Example: 
        start_date=0, end_date=100, prediction_start=100, prediction_end=101
        The data X is 100 days and the prediction y is 1 day irrespective if the start day is 
        0 -> 100, 
        200 -> 300, 
        217 -> 317 
        etc.
        ...

        Args:
        start_date: int, the start date of the data X in days starting from 0
        end_date: int, the end date of the data X in days

        #between end_date and prediction_start can be a gap

        prediction_start: int, the start date of the prediction y in days
        prediction_end: int, the end date of the prediction y in days

        random_sample: int, the number of random samples to be taken from the dataset

        Returns:
        X_dataset: pd.DataFrame, the data X dataset
        y_dataset: pd.Series, the prediction y dataset
        """

        input_time = end_date-start_date
        gap = prediction_start-start_date
        prediction_time = prediction_end-start_date

        if not random_sample:  
            r = range(0,self.data[0].shape[0]-(prediction_end-start_date)+1)
        else:
            r = random.sample(range(0,self.data[0].shape[0]-(prediction_end-start_date)+1),random_sample)
        
        for i, day in enumerate(r):
            print(f'Day: {i+1} of {len(r)}', end='\r')
            for tree in range(self.__len__()):
                #print(f'Tree {tree}', end='\r')
                X, y = self.get_sample(day, day+input_time, day+gap, day+prediction_time, tree)
                if y is not None:
                    
                    if i == 0:
                        X_dataset, columns_X = X.values.reshape(1,-1), list(X.index)
                        y_dataset, columns_y = y.values.reshape(1,-1), list(y.index)
                    else:
                        X_dataset = np.concatenate([X_dataset,X.values.reshape(1,-1)], axis=0)
                        y_dataset = np.concatenate([y_dataset,y.values.reshape(1,-1)], axis=0)


        return pd.DataFrame(X_dataset,columns=columns_X), pd.DataFrame(y_dataset, columns=columns_y)
    
    def get_sample(self,start_date, end_date, prediction_start, prediction_end, tree=None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Function which retrieves one sample from the dataset with the given start and end date for the data X 
        and the prediction_start and prediction_end for the prediction y.
        Exact dates in the data.

        Args:
        start_date: int, the start date of the data X in days starting from 0 
        end_date: int, the end date of the data X in days

        #between end_date and prediction_start can be a gap

        prediction_start: int, the start date of the prediction y in days
        prediction_end: int, the end date of the prediction y in days

        tree: int, the tree from which the sample should be retrieved

        Returns:
        X: pd.DataFrame, the data X sample
        y: pd.Series, the prediction y sample
        """
        prediction_end = prediction_start+1 #if prediction_start==prediction_end else NotImplementedError('The prediction_end must be one day after the prediction_start not yet implemented. Or sum up the days in the prediction_start')
        tree = random.randint(0,self.__len__()-1) if not tree else tree
        X = self.retrieve_variables(self.data[tree].iloc[start_date:end_date, :])
        #y = self.one_hot_encoding_flowers(self.data[tree].iloc[prediction_start:prediction_end, :].loc[:,['FemaleInflo','MaleInflo']].sum()) #for 20 day periods
        y = self.one_hot_encoding_flowers(self.data[tree].iloc[prediction_start:prediction_end, :].loc[:,['FemaleInflo_r','MaleInflo_r']]) #daily
        
        return X, y
    
    def retrieve_variables(self,X:pd.DataFrame) -> pd.Series:
        """
        Function which determines the INPUT variables of X sample-wise for the Random Forest

            Args:
            X: pd.DataFrame, the orginal data to be transformed

            Returns:
            pd.Series, the transformed data
        """
        series_dict = {}
        for time in range(0,len(X)-100+1,100):
            for var in ['HRMin', 'HRMax', 'HRAverage']:
                for magnitude in range(0,91,10):
                    series_dict[f'{var}_{time}_{magnitude}'] = sum(1 for x in X.iloc[time:time+101,:][var] if magnitude < x < magnitude + 10)
            for var in ['TMin', 'TMax', 'TAverage']:
                for magnitude in range(20,51,10):
                    series_dict[f'{var}_{time}_{magnitude}'] = sum(1 for x in X.iloc[time:time+101,:][var] if magnitude < x < magnitude + 10)

            # all events which only need to be counted occur here
            series_dict[f'RankOneLeafDate_{time}'] = sum(X.iloc[time:time+101,:]['RankOneLeafDate'])
            series_dict[f'OpenedSpatheDate_compl_{time}'] = sum(X.iloc[time:time+101,:]['OpenedSpatheDate_compl'])
            series_dict[f'FemaleInflo_{time}'] = sum(X.iloc[time:time+101,:]['FemaleInflo'])
            series_dict[f'MaleInflo_{time}'] = sum(X.iloc[time:time+101,:]['MaleInflo'])
        
        return pd.Series(series_dict)
    
    def one_hot_encoding_flowers(self,y:pd.DataFrame) -> pd.Series:
        """
        Function which one-hot encodes the flowers for the Random Forest
        Args: 
        y: pd.DataFrame, the orginal data to be transformed
        
        Returns:
        pd.Series, the transformed data as classes"""
        ohe_flowers = []
        for flower in range(len(y)):
            if y['FemaleInflo_r'].iloc[flower] == y['MaleInflo_r'].iloc[flower] and y['FemaleInflo_r'].iloc[flower] == 0:
                ohe_flowers.append(0)
            #no flowers
            if y['FemaleInflo_r'].iloc[flower] == y['MaleInflo_r'].iloc[flower] and y['FemaleInflo_r'].iloc[flower] != 0:
                ohe_flowers.append(1)
            #more likely to have females
            elif y['FemaleInflo_r'].iloc[flower] > y['MaleInflo_r'].iloc[flower]:
                ohe_flowers.append(2)
            #more likely to have males
            elif y['FemaleInflo_r'].iloc[flower] < y['MaleInflo_r'].iloc[flower]:
                ohe_flowers.append(3)
        
        # #no flowers
        # if y['FemaleInflo_r'] == y['MaleInflo_r'] and y['FemaleInflo_r'] == 0:
        #     ohe_flowers.append(0)
        # #male and female equally likely
        # elif y['FemaleInflo_r'] == y['MaleInflo_r'] and y['FemaleInflo_r'] != 0:
        #     ohe_flowers.append(1)
        # #more likely to have females
        # elif y['FemaleInflo_r'] > y['MaleInflo_r']:
        #     ohe_flowers.append(2)
        # #more likely to have males
        # elif y['FemaleInflo_r'] < y['MaleInflo_r']:
        #     ohe_flowers.append(3)
            
            

        return pd.Series(ohe_flowers, name='flowers')


# %%
dataset = RandomForestDataset(meteo_data=meteo,pheno_data=pheno)

# %%
dataset.get_sample(0,100,100,120)[1]

# %%
d = dataset.get_dataset(0, 700, 1000, 1020)
d[0].to_csv('/home/u108-n256/PalmProject/RandomForest/20240916_X',index=False)
d[1].to_csv('/home/u108-n256/PalmProject/RandomForest/20240916_y',index=False)

'''
from collections import Counter
import pickle

Counter(d[1].values.reshape(-1))



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(d[0],d[1],test_size=0.1)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 25, 30, None],
    'min_samples_split': [5, 10, 20, 30],
    'min_samples_leaf': [5, 10, 20, 30]
}

rfc = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

# %%
#grid_search.best_params_

# %%
rfc = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=5, min_samples_split=10, max_features='sqrt')
rfc.fit(x_train, y_train)

# %%
print(rfc.estimators_[0].max_depth)

# %%
predict = rfc.predict(x_test)
'''
