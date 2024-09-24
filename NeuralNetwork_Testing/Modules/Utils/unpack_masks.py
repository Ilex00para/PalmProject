import os
import numpy as np

def unpack_masks(masks: list[str], directory='/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs/NN_Inputs_masks'):
    mask_list = []
    if masks[0] is None:
        return mask_list.append(None)
    for m in masks:
        mask_path = os.path.join(directory, m)
        mask_list.append(np.load(mask_path))
    return mask_list