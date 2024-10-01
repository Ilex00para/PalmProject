# Random Forest

### Data
Uses raw meteorological and phenological data from each site, sourced from `NeuralNetwork_Testing/NN_Inputs/SMSE/RAW_Meteorological_Data.npy` and `RAW_Phenological_Data.npy`. Data handling is managed by the `RandomForestDataset()` object.

### Classification (y)
The random forest predicts the sex of the flower on a given day: male, female, or no flowers. Classes are assigned by smoothing daily male and female flower observations using a Gaussian distribution kernel or the mean of 40 days. The more probable sex is assigned to each day. Days with equal probabilities are excluded from training to force model decisions. If no flower probability is present, the class is "no flowers". This process is performed by `one_hot_encoding_flowers(self, y: np.array)` in `RandomForestDataset()` (though it is not true one-hot encoding).

### Input (X)
Input data uses extreme weather events. Quantile borders for each variable (temperature, relative humidity, rainfall) were calculated to determine the highest 5%, highest 25%, lowest 5%, and lowest 25% of events. The input time frame covers 700 days (modifiable in `get_dataset(self, start_date, end_date, prediction_start, random_sample: int = None)`), split into 7 segments of 100 days. The number of extreme events per segment was summed and used as an input variable.

**Additional X Information** could be added and includes summed phenological variable values for each segment, added to the meteorological input data as `[meteo input array] + [pheno input array] = X`. Different X combinations were managed across scripts. But the `RandomForestDataset()` allows multiple datasets through subclasses like `RandomForestDatasetQuantile(RandomForestDataset)` by overwriting the `retrieve_variables(self, X: np.array)` function.

#### Additional info
`m&f`in file names = male and female flowers in input 
`RandomForest_extremeEvents&BunchLoad.ipynb` is most elaborated and commented.