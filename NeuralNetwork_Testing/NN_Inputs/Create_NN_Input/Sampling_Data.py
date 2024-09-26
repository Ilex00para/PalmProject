import numpy as np


def days_to_features(Meteorological_Data: np.array, X: int = 20) -> np.array:
    """
    Transforms daily meteorological features into aggregated features over X-day periods.

    This function reshapes and aggregates the daily meteorological data into features representing 
    data over X-day periods. The input data is assumed to be a 4D numpy array, where the dimensions 
    represent trees, samples, days, and features respectively.

    Parameters:
    Meteorological_Data (np.array): A 4D numpy array with dimensions (trees, samples, days, features).
    X (int): The number of days to aggregate into one period. Default is 20.

    Returns:
    np.array: A 4D numpy array with transformed dimensions (trees, samples, 40, 180), 
              where 40 represents the number of X-day periods and 180 is the total number of features 
              after reshaping each X-day period.
    """
    # Check if the input data has the expected shape
    if len(Meteorological_Data.shape) != 4:
        raise ValueError("Meteorological_Data must be a 4D numpy array with dimensions (trees, samples, days, features).")

    # Initialize the array to store the transformed data
    transposed_array = np.empty((Meteorological_Data.shape[0], Meteorological_Data.shape[1], 50, 180))

    # Iterate over each tree
    for tree in range(Meteorological_Data.shape[0]):
        # Initialize array to store transformed data for each sample of the current tree
        tree_array = np.empty((Meteorological_Data.shape[1], 50, 180))
        
        # Iterate over each sample for the current tree
        for sample in range(Meteorological_Data.shape[1]):
            # Initialize tensor to store transformed data for each X-day period
            sample_tensor = np.empty((50, 180))
            
            # Iterate over each X-day period
            for i, day in enumerate(range(0, Meteorological_Data.shape[2], X)):
                # Extract the data for the current X-day period and reshape it
                new_day = Meteorological_Data[tree, sample, day:day+X, :]
                sample_tensor[i-1, :] = new_day.reshape((1, 180))
            
            # Store the transformed sample tensor in the tree array
            tree_array[sample, :, :] = sample_tensor
        
        # Store the transformed tree array in the final transposed array
        transposed_array[tree, :, :] = tree_array
    
    return transposed_array


def sample_meteorological_data(Meteorological_Data: np.array, tree_number: int,  X: int =20) -> np.array:
    """
    Create samples of X-day periods from the meteorological data.
    
    args.
    Meteorological_Data: np.array - The meteorological data to sample.
    tree_number: int - The number of trees in the data for repeating the meteo data.
    X: int - The number of days to aggregate into one period. Default is 20.

    return
    Meteorological_Data_Samples: np.array - The samples of X-day periods.
    """
    # Calculate the maximum number of usable days that can be divided evenly into 20-day periods
    useable_days = int(Meteorological_Data.shape[0] / X) * X
    
    # Initialize an empty array to hold the samples
    # Each sample consists of 900 days of data
    # The shape of Meteorological_Data_Samples is (number of samples, 900, number of features)
    Meteorological_Data_Samples = np.zeros((int((useable_days - 1000) / X) + 1, 1000, Meteorological_Data.shape[1]))
    
    
    # Loop through the Meteorological_Data array, creating samples of 900 consecutive days,
    # starting every 20 days
    for sample_idx, i in enumerate(range(0, (useable_days - 1000) + 1, X)):
        # Extract a sample of 900 days and assign it to the appropriate position in Meteorological_Data_Samples (0-900, 20-920, 40-940, ...)
        Meteorological_Data_Samples[sample_idx,:,:] = Meteorological_Data[i:i + 1000, :]

    Meteorological_Data_Samples = np.repeat(np.expand_dims(Meteorological_Data_Samples, axis=0), tree_number,axis=0)
    Meteorological_Data_Samples = days_to_features(Meteorological_Data_Samples)
    # Return the array of samples
    return Meteorological_Data_Samples

def sampeling_phenological_data(Phenological_Data: np.array, X: int =20) -> np.array:
    # Create samples of 45 x 20 days = 900 days in total, every 5 days
    # prediction time 2 x 20 = 40 days, gap 8 x 20 = 160 days and 35 x 20 = 700 days
    # (trees = 266 x months to use (175-45), days = 45, features = 10) --> Phenological_Data_Samples
    total_time = int(1000/X)
    # Initialize an empty array to hold the samples
    # The shape of Phenological_Data_Samples is (number of samples, 45 days, number of features)
    Phenological_Data_Samples = np.empty((Phenological_Data.shape[0] * (Phenological_Data.shape[1] - total_time), total_time, Phenological_Data.shape[2]))
    
    # Initialize a counter for the sample index
    sample_idx = 0
    
    # Loop through each "tree" in the Phenological_Data array
    for tree in range(Phenological_Data.shape[0]):
        # For each tree, loop through the time steps, creating samples of 45 days
        for i in range(0, (Phenological_Data.shape[1] - total_time)):
            # Extract a sample of 45 days and assign it to the appropriate position in Phenological_Data_Samples
            Phenological_Data_Samples[sample_idx] = Phenological_Data[tree, i:i + total_time, :]
            # Increment the sample index
            sample_idx += 1
    
    Phenological_Data_Samples = Phenological_Data_Samples.reshape(Phenological_Data.shape[0], Phenological_Data.shape[1]-total_time, total_time, Phenological_Data.shape[2])

    # Return the array of samples
    return Phenological_Data_Samples

def merge_samples_meteo_pheno(Phenological_Data, Meteorological_Data, X: int =20):
        """Args
        Phenological_Data: np.array
        Meteorological_Data: np.array
        X: int - The number of days to aggregate into one period. Default is 20.

        return
        Complete_Samples: np.array
        """
        Phenological_Data_Samples = sampeling_phenological_data(Phenological_Data, X)
        Meteorological_Data_Samples = sample_meteorological_data(Meteorological_Data,Phenological_Data.shape[0],X)
        if Phenological_Data_Samples.shape[:-1] == Meteorological_Data_Samples.shape[:-1]:
            Complete_Samples = np.concatenate((Phenological_Data_Samples, Meteorological_Data_Samples), axis=-1)
            return Complete_Samples
        else:
            raise ValueError(f"The number of samples in the phenological and meteorological data must be the same.\nWith shape {Phenological_Data_Samples.shape} and {Meteorological_Data_Samples.shape[:-1]} respectively.")


