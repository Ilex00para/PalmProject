# `full_training` Function Documentation

Purpose:

The full_training function trains a machine learning model using the provided training and testing dataloaders, optimizer, and loss function over a specified number of epochs. The function supports optional learning rate scheduling, gradient tracking, verbosity, and early stopping.

Parameters:

model (torch.nn.Module): The neural network model to be trained.

trainining_dataloader (torch.utils.data.DataLoader): DataLoader providing the training data batches.

testing_dataloader (torch.utils.data.DataLoader): DataLoader providing the testing/validation data batches.

optimizer (torch.optim.Optimizer): Optimizer to be used for training the model.
loss_function (callable): Loss function to compute the loss between predicted and actual values.

epochs (int, optional): Number of epochs for training the model. Default is 100.

device (str, optional): Device to train the model on. Default is 'cuda' for GPU training.

lr_scheduler (optional): Learning rate scheduler to adjust the learning rate during training. Default is None.

kwargs: Additional keyword arguments:

gradients (bool): If set to True, gradients of the model parameters will be computed and tracked.

verbose (bool): If set to True, training progress and metrics will be printed at each epoch.

early_stopping (bool): If set to True, early stopping will be applied if the validation loss does not improve over a certain number of epochs (50 epochs in this case).

Returns:

model (torch.nn.Module): The trained model.

training_progress (pd.DataFrame): DataFrame containing the training progress with columns ['Epoch', 'Loss', 'Validation Loss', 'RMSE'].

predictions (pd.DataFrame): DataFrame containing the actual and predicted values with columns ['Actual', 'Predicted'].

Detailed Description:
Model Initialization:

The model is moved to the specified device (GPU or CPU).
Training Loop:

The model is trained for the specified number of epochs.
For each epoch:
The training dataloader is iterated over, and for each batch, the training step is performed using the Train_Step function.
The testing dataloader is iterated over, and for each batch, the validation step is performed using the Test_Step function.
If the gradients option is enabled, gradients of the model parameters are computed.
If the verbose option is enabled, the current epoch, loss, validation loss, and RMSE are printed.
On the first epoch, DataFrames for tracking training progress and predictions are initialized.
Training progress and predictions are updated using the track_metrics function.
If the early_stopping option is enabled, training is stopped early if the validation loss does not improve over the last 50 epochs.
Metric Tracking and Early Stopping:

Training progress and predictions are tracked and updated at each epoch.
Early stopping is applied based on the validation loss trend if the early_stopping option is enabled.
Final Output:

The function returns the trained model, training progress DataFrame, and predictions DataFrame.