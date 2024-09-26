import numpy as np

def time_unit_reduction(Phenological_Data: np.array, X: int = 20) -> np.array:
    """
    Reduce daily phenological data to X-day aggregated data.

    This function aggregates the daily phenological data into periods of X days. For most features,
    it sums the values over each X-day period. For the bunchload feature (assumed to be at index 9),
    it computes the mean over each X-day period.

    Parameters:
    Phenological_Data (np.array): A 3D numpy array where each element represents data for a tree, 
                                  day, and feature.
    X (int): The number of days to aggregate. Default is 20.

    Returns:
    np.array: A new numpy array with the aggregated data.
    """
    # Calculate the shape of the new array: (number of trees, number of X-day periods, number of features)
    new_shape = (Phenological_Data.shape[0], (Phenological_Data.shape[1] // X) + 1, Phenological_Data.shape[2])
    aggregated_data = np.empty(new_shape)

    # Iterate over each tree
    for tree in range(Phenological_Data.shape[0]):
        # Iterate over each X-day period
        for i, days in enumerate(range(0, Phenological_Data.shape[1], X)):
            # Iterate over each feature
            for feature in range(Phenological_Data.shape[2]):
                if feature != 9:
                    # Sum the feature values over the X-day period
                    aggregated_data[tree, i, feature] = np.sum(Phenological_Data[tree, days:days+X, feature], axis=0)
                else:
                    # Compute the mean for the bunchload feature at index 9 over the X-day period
                    chunk = np.sum(Phenological_Data[tree, days:days+X, feature], axis=0)
                    aggregated_data[tree, i, feature] = chunk / X

    return aggregated_data

def apply_Moving_Average(Phenological_Data: np.array, add = False) -> np.array:
    """
    Apply a moving average to the phenological data of flowers.

    This function calculates a moving average over a window of 3 periods for male, female, 
    and aborted flowers data and updates the original Phenological_Data array accordingly.

    Parameters:
    Phenological_Data (np.array): A 3D numpy array where each element represents data for a tree,
                                  period, and flower type (6: female, 7: male, 8: aborted).

    Returns:
    np.array: The updated Phenological_Data array with applied moving averages.
    """
    # Initialize arrays to store moving averages for each flower type
    male_MA_flowers = np.zeros((Phenological_Data.shape[1], 1))
    female_MA_flowers = np.zeros((Phenological_Data.shape[1], 1))
    aborted_MA_flowers = np.zeros((Phenological_Data.shape[1], 1))

    # Loop through each tree
    for tree in range(Phenological_Data.shape[0]):
        # Loop through each period of 20 days
        for period_20_days in range(Phenological_Data.shape[1]):
            # Calculate moving average for the beginning of the array
            if period_20_days == 0:
                ma_f = np.sum(Phenological_Data[tree, period_20_days:period_20_days+2, 6], axis=0)
                ma_m = np.sum(Phenological_Data[tree, period_20_days:period_20_days+2, 7], axis=0)
                ma_a = np.sum(Phenological_Data[tree, period_20_days:period_20_days+2, 8], axis=0)
            # Calculate moving average for the end of the array
            elif period_20_days == Phenological_Data.shape[1]-1:
                ma_f = np.sum(Phenological_Data[tree, period_20_days-1:period_20_days, 6], axis=0)
                ma_m = np.sum(Phenological_Data[tree, period_20_days-1:period_20_days, 7], axis=0)
                ma_a = np.sum(Phenological_Data[tree, period_20_days-1:period_20_days, 8], axis=0)
            # Calculate moving average for the middle of the array
            else:
                ma_f = np.sum(Phenological_Data[tree, period_20_days-1:period_20_days+2, 6], axis=0)
                ma_m = np.sum(Phenological_Data[tree, period_20_days-1:period_20_days+2, 7], axis=0)
                ma_a = np.sum(Phenological_Data[tree, period_20_days-1:period_20_days+2, 8], axis=0)

            # Store the moving average values
            female_MA_flowers[period_20_days, :] = ma_f
            male_MA_flowers[period_20_days, :] = ma_m
            aborted_MA_flowers[period_20_days, :] = ma_a
        
        if add:
            Phenological_Data = np.concatenate((Phenological_Data, female_MA_flowers.reshape(-1), male_MA_flowers.reshape(-1), aborted_MA_flowers.reshape(-1)), axis=2)

        # Update the original Phenological_Data array with the calculated moving averages
        '''
        Phenological_Data[tree, :, 6] = female_MA_flowers.reshape(-1)
        Phenological_Data[tree, :, 7] = male_MA_flowers.reshape(-1)
        Phenological_Data[tree, :, 8] = aborted_MA_flowers.reshape(-1)
        '''

    return Phenological_Data

def process_phenological_data(Phenological_Data: np.array, X: int = 20, apply_moving_average = True) -> np.array:
    """
    Reduces the input 3D array along its second dimension in chunks of 20.
    For each chunk of 20 days, it computes the sum of all features except the 
    feature at index 9, for which it computes the mean.

    Parameters:
    Phenological_Data (np.array): A 3D numpy array where the dimensions are 
                             (number of trees, number of days, number of features).

    Returns:
    np.array: A 3D numpy array reduced along the second dimension.
    """

    Phenological_Data = time_unit_reduction(Phenological_Data, X)

    if apply_moving_average:
        Phenological_Data = apply_Moving_Average(Phenological_Data)
     
    return Phenological_Data