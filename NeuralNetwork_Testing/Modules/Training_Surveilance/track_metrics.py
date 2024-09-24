import pandas as pd

def track_metrics(epoch: int, loss: float, validation_loss: float, metrics: tuple,training_progress: pd.DataFrame, predictions: pd.DataFrame):
    """
    Track and update training and prediction metrics.

    This function updates the training progress and prediction dataframes with the provided metrics for the given epoch. 
    If the epoch is 0, it initializes the dataframes with appropriate columns.

    Parameters:
    -----------
    training_progress : pd.DataFrame
        DataFrame to track the training progress including loss, validation loss, and RMSE for each epoch.
    predictions : pd.DataFrame
        DataFrame to track the actual and predicted values.
    epoch : int
        The current epoch number.
    loss : float
        The training loss for the current epoch.
    validation_loss : float
        The validation loss for the current epoch.
    metrics : tuple
        A tuple containing RMSE (float), actual values (array-like), and predicted values (array-like).

    Returns:
    --------
    training_progress : pd.DataFrame
        Updated DataFrame with the current epoch's training progress.
    predictions : pd.DataFrame
        Updated DataFrame with the current epoch's predictions.
    """
    # Create a dictionary for the current epoch's training metrics
    epoch_dict = {'Epoch': epoch, 'Loss': loss, 'Validation Loss': validation_loss, 'RMSE': metrics[0]}
    # Append the dictionary to the training progress dataframe
    training_progress = pd.concat([training_progress, pd.Series(epoch_dict).to_frame().T], ignore_index=True) if not training_progress.empty else pd.DataFrame(epoch_dict, index=[0])

    # Create a dictionary for the current epoch's predictions
    prediction_dict = {'Actual': metrics[1], 'Predicted': metrics[2]}
    # Append the dictionary to the predictions dataframe
    predictions = pd.concat([predictions, pd.Series(prediction_dict).to_frame().T], ignore_index=True) if not training_progress.empty else pd.DataFrame(prediction_dict, index=[0])

    return training_progress, predictions
