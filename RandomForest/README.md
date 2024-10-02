# Random Forest

## Context

The random classifier was developed as part of the **CIGE oil palm project**. The **CIGE oil palm project** is an initiative under the broader research efforts of **CIRAD** (Centre de coopération Internationale en Recherche Agronomique pour le Développement), which is a French agricultural research organization focused on fostering sustainable agricultural practices in developing countries.

The **CIGE project** aims to improve the understanding of **genetic and environmental interactions** (GxE) in oil palm cultivation. Specifically, it focuses on optimizing the productivity and sustainability of oil palm plantations, addressing the critical need for more efficient and environmentally friendly practices in the face of climate change and deforestation concerns. The research within the project emphasizes the following:

1. **Genetic Improvement**: Developing improved varieties of oil palm that are more resistant to biotic (pests and diseases) and abiotic stresses (like drought and poor soils). This helps in enhancing yield while reducing the reliance on expanding plantation areas.

2. **Environmental Assessment**: Understanding the interactions between various oil palm varieties and their growing conditions, which includes analyzing the effects of different soil types, weather patterns, and agronomic practices. This helps optimize the best variety for specific environments, thereby improving resource use efficiency.

3. **Sustainable Practices**: The project also focuses on promoting **sustainable intensification** of oil palm. By studying both genetics and agronomic management, the goal is to increase productivity without additional land conversion, aligning with broader goals of reducing deforestation and conserving biodiversity.

The random forest is supossed to forecast the sex of a flower on a given day in the future based on information of climate and physiology. The data used for training were collected and documented by CIRAD Researchers on different locations (Indonesia SMSE, Nigeria PR). The random forest gives also insights to factors which affect the determination of the flower sex.

### Data
Uses raw meteorological and phenological data from each site, sourced from `NeuralNetwork_Testing/NN_Inputs/SMSE/RAW_Meteorological_Data.npy` and `../RAW_Phenological_Data.npy`. Data handling is managed by the `RandomForestDataset()` object.

### Classification (y)
The random forest predicts the sex of the flower on a given day: male, female, or no flowers. Classes are assigned by smoothing daily male and female flower observations using a Gaussian distribution kernel or the mean of 40 days, capturing trend instead of point observations. The more probable sex is assigned to each day. Days with equal probabilities are excluded from training to force model decisions. If no flower probability is present, the class is "no flowers". This process is performed by `one_hot_encoding_flowers(self, y: np.array)` in `RandomForestDataset()` (though it is not true one-hot encoding).

### Input (X)
Input data uses extreme weather events. Quantile borders for each variable (temperature, relative humidity, rainfall) were calculated to determine the highest 5%, highest 25%, lowest 5%, and lowest 25% of events. The quantile border determines above or below whoch threshold a value of a variable needs to be considered an extreme (25%) or very extreme (5%) event. The extreme events were choosen because it was assumed that they have significant effects on flower development. The input time frame covers 700 days (modifiable in `get_dataset(self, start_date, end_date, prediction_start, random_sample: int = None)`), split into 7 segments of 100 days this segmentation allows the model to learn how different conditions over time impact flowering outcomes. The number of extreme events per segment was summed and used as an input variable.

**Additional X Information** could be added and includes summed phenological variable values for each segment, added to the meteorological input data as `[meteo input array] + [pheno input array] = X`. Different X combinations were managed across scripts. But the `RandomForestDataset()` allows multiple datasets through subclasses like `RandomForestDatasetQuantile(RandomForestDataset)` by overwriting the `retrieve_variables(self, X: np.array)` function.

### Training/ Fitting
Before fitting the random forest, instances of class (4) "equal chance of male & female" were removed from the dataset. The dataset was then split into an 80/20 ratio for training and validation. To address class imbalance, the training dataset was upsampled by duplicating random samples from underrepresented classes until all three classes were equally represented. The random forest was subsequently trained on this upsampled dataset. Although hyperparameter optimization was not performed, the script includes a section for hyperparameter grid search, which can be uncommented if needed.

### Feature Importance
The decision trees in the random forest were analyzed using the `compute_feature_importances()` function from scikit-learn to determine the influence of each feature on the model's decisions. The importance scores from all trees were aggregated for every feature to identify those most frequently used in decision-making. For example, if relative humidity or temperature extremes consistently receive high importance scores, it suggests that these variables are critical factors in determining flower sex. This feature importance analysis provides valuable insights into which climatic and physiological variables have the most significant impact on flower development, guiding future research and optimization efforts.

#### Additional info
`m&f`in file names = male and female flowers in input 
`RandomForest_extremeEvents&BunchLoad.ipynb` is most elaborated and commented file.