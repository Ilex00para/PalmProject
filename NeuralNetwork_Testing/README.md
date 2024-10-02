## Modules
Modular cide used during the training, prediciton and analysis (shapley) of the CNN. The Main Scripts run several processes from Grid Search of the Architecture (hyper)parameters, Training of the NN, the cross validation of different inputs (X), the prediction of trained models and the analysis of the architectures with shapley values. The main scripts use functions or classes of the other folders in Modules like the Architectures and the code for the datasets as pytorch dataset classes.

## NN_Inputs
Creates Neural Network Input data from the `dataCIGE` folder (which is not in this repository) with the `Create_NN_InputData.py` file. Different inputs for the same architecture are used by masking the input data. The masks are created and stored in the `NN_Inputs_masks` folder. If more then two sites are used a global scaling can be applied (more specified in the NN_Inputs README)

## dataCIGE
Empty folder which contains the data used by `NN_Inputs`. Needs to be filled.
