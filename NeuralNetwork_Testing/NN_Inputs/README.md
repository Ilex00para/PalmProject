# NN_INPUTS

`Create_NN_Input` contains funtions used in `Create_NN_InputData.py`

`NN_Input_masks` contains masks and the code to create them for masking out the input features of the model.

`Create_NN_InputData.py` is the main script used to bring the raw data from the `NeuralNetwork_Testing/dataCIGE` folder into a format which can be used by NNs.


Is created:

`PR`,`SMSE`, etc. are the folders containing the data for each of the sites 

**!!! If the sites should be included in the global scaling this needs to be done in the `NeuralNetwork_Testing/NN_Inputs/Create_NN_Input/MinMax_Scaling.py` script!!!**

`Scaler_Meteorological.pkl` & `Scaler_Phenological.pkl` pickled global max-min scaler