import numpy as np

def normalize_array(arr,axis=None):
    """
    Normalize a numpy array to the range [0, 1].

    Parameters:
    arr (numpy.ndarray): The array to normalize.

    Returns:
    numpy.ndarray: The normalized array.
    """
    assert len(arr.shape) < 3, 'Array should be 1D or 2D'
    
    if axis == None:
        min_value = np.min(arr)
        max_value = np.max(arr)
        try:
            normalized_arr = (arr - min_value) / (max_value - min_value)
        except ZeroDivisionError:
            print(f'Minimum Value {min_value}, Maximum Value {max_value}')
        return normalized_arr
    
    if axis == 0:
        normalize_arr = np.zeros_like(arr)
        for col in range(arr.shape[0]):
            normalize_arr[col] = (arr[col,:] - np.min(col)) / (np.max(arr) - np.min(col))
        return normalize_arr
    
    if axis == 1:
        normalize_arr = np.zeros_like(arr)
        for row in range(arr.shape[1]):
            normalize_arr[row] = (arr[:,row] - np.min(row)) / (np.max(arr) - np.min(row))
        return normalize_arr