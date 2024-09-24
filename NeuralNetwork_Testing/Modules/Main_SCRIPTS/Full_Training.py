from Modules.Training_Surveilance import track_metrics, get_gradients, early_stopping
from Modules.Utils import Train_Step, Test_Step
import numpy as np
import pandas as pd



def full_training(model, trainining_dataloader, testing_dataloader, optimizer, loss_function, epochs: int = 100, device: str = 'cuda', lr_scheduler=None, **kwargs):
    """
    Trains a machine learning model using provided dataloaders, optimizer, and loss function over a specified number of epochs.
    
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
        early_stopping (bool): If set to True, early stopping will be applied if the validation loss does not improve over a certain number of epochs.

    Returns:
    model (torch.nn.Module): The trained model.
    training_progress (pd.DataFrame): DataFrame containing the training progress with columns ['Epoch', 'Loss', 'Validation Loss', 'RMSE'].
    predictions (pd.DataFrame): DataFrame containing the actual and predicted values with columns ['Actual', 'Predicted'].
    """
    
    # Move model to the specified device (GPU or CPU)
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        for batch in trainining_dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            loss = Train_Step(model=model, x=X, y=y, optimizer=optimizer, loss_function=loss_function, lr_scheduler=lr_scheduler)

        # Validation phase
        for batch in testing_dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            validation_loss, metrics = Test_Step(model=model, x=X, y=y, loss_function=loss_function)
        
        "Following code is not necessary for the training but it is useful for tracking the training progress"
        # Convert loss and validation loss to float for tracking
        loss, validation_loss = float(loss.detach().to('cpu').numpy()), float(validation_loss.detach().to('cpu').numpy())
        
        # Initialize DataFrames for tracking metrics and predictions at the first epoch
        if epoch == 0:
            training_progress,predictions = pd.DataFrame(columns=['Epoch', 'Loss', 'Validation Loss', 'RMSE']), pd.DataFrame(columns=['Actual', 'Predicted'])

        # Track training progress and predictions in pd.Dataframes
        training_progress, predictions = track_metrics(training_progress=training_progress, 
                                                       predictions=predictions, 
                                                       epoch=epoch, 
                                                       loss=loss, 
                                                       validation_loss=validation_loss, 
                                                       metrics=metrics)
        
        # Compute and track gradients
        if kwargs.get('gradients'):
            get_gradients(model)
        
        # Print training progress if verbosity is enabled
        if kwargs.get('verbose'):
            print(f'Epoch: {epoch}, Loss: {loss}, Validation Loss: {validation_loss}, RMSE: {metrics[0]}, LR: {optimizer.param_groups[0]["lr"]}')
        
        #saving the model if best so far
        if epoch == 0:
            best_model = model.state_dict().copy()
        if training_progress['Validation Loss'].iloc[-1] < training_progress['Validation Loss'].iloc[:-1].min():
            best_model = model.state_dict().copy()
            print('Best model updated')
            
        # Early stopping based on the pd.Dataframe of the training progress
        if kwargs.get('early_stopping'):
            if early_stopping(training_progress, threshold=50):
                return model, best_model, training_progress.dropna(axis=0), predictions.dropna(axis=0)
    
    return model, best_model, training_progress.dropna(axis=0), predictions.dropna(axis=0)