import numpy as np
import os
import numpy as np
from typing import List, Tuple

index_feature_dict ={
    0 : 'RankOneLeafDate',
    1 : 'AppearedSpatheDate_compl', 
    2 : 'OpenedSpatheDate_compl', 
    3 : 'FloweringDate_compl', 
    4 : 'HarvestDate_compl', 
    5 : 'BunchMass', 
    6 : 'FemaleInflo', 
    7 : 'MaleInflo', 
    8 : 'AbortedInflo', 
    9 : 'BunchLoad',
    10 :'TMin',
    11 :'TMax',
    12 :'TAverage',
    13 :'HRMin',
    14 :'HRMax',
    15 :'HRAverage',
    16 :'WindSpeed',
    17 :'Rainfall',
    18 :'Rg'
    }

# Create a mask to exclude certain features from the data

def create_mask(index: List[int], size: Tuple[int, int] = (50, 190)) -> np.ndarray:
    """
    Create a boolean mask array of the given size with specified columns set to False.

    Args:
        index (List[int]): A list of column indices to be masked.
        size (Tuple[int, int], optional): The size of the mask array. Default is (50, 190).

    Returns:
        np.ndarray: A boolean mask array with specified columns set to False.

    Raises:
        AssertionError: If any index is not in the range 0 to 18.

    Example:
        >>> mask = create_mask([1, 12], (50, 190))
        >>> mask.shape
        (50, 190)
        >>> mask[:, 1].all()
        False
        >>> mask[:, 108].all()
        False
    """
    # Initialize the mask array with True values
    mask = np.ones((size[0], size[1]), dtype=bool)
    
    # Ensure that all indices are within the valid range
    assert all(0 <= i <= 18 for i in index), 'Invalid index'
    
    # Iterate over the list of indices to set corresponding columns to False
    for i in index:
        if i < 10:
            # For indices less than 10, set the entire column to False
            mask[:, i] = False
        elif 10 <= i < 19:
            print(f'i: {i}')
            # For indices from 10 to 18, set specific columns to False
            for j in range(0, 20):
                col_index = (j * 9) + i
                mask[:, col_index] = False
    return mask


cwd = os.getcwd()

for index in index_feature_dict.keys():
    mask = create_mask([index])
    np.save(os.path.join(cwd, f'NeuralNetwork_Testing/NN_Inputs/NN_Inputs_masks/{index_feature_dict[index]}_mask.npy'), mask)
    print(f'{index_feature_dict[index]} mask created')

meteo_mask = create_mask([0,1,2,3,4,5,6,7,8,9])
np.save(os.path.join(cwd, 'NeuralNetwork_Testing/NN_Inputs/NN_Inputs_masks/Meteorological_mask.npy'), meteo_mask)

pheno_mask = create_mask([10,11,12,13,14,15,16,17,18])
print(pheno_mask)
np.save(os.path.join(cwd, 'NeuralNetwork_Testing/NN_Inputs/NN_Inputs_masks/Phenological_mask.npy'), pheno_mask)

bunchload_mask = create_mask([0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18])
np.save(os.path.join(cwd, 'NeuralNetwork_Testing/NN_Inputs/NN_Inputs_masks/BunchLoad_only_mask.npy'), bunchload_mask)

meteo_and_bunchload_mask = create_mask([0,1,2,3,4,5,6,7,8])
np.save(os.path.join(cwd, 'NeuralNetwork_Testing/NN_Inputs/NN_Inputs_masks/Meteo_and_Bunchload_mask.npy'), meteo_and_bunchload_mask)

meteo_and_flowers_mask = create_mask([0,1,2,3,4,5,8,9])
np.save(os.path.join(cwd, 'NeuralNetwork_Testing/NN_Inputs/NN_Inputs_masks/Meteo_and_Flower_mask.npy'), meteo_and_flowers_mask)
