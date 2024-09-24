import pandas as pd
def early_stopping(training_progress: pd.DataFrame, threshold: int = 50) -> bool:
    """
    Check if early stopping should be applied based on validation loss.

    Args:
        training_progress (pd.DataFrame): A DataFrame containing the training progress, 
                                          including at least a 'Validation Loss' column.
        threshold (int, optional): The number of epochs to check for improvement. Default is 50.

    Returns:
        bool: True if early stopping should be applied, False otherwise.
    """
    # Get the index of the minimum validation loss
    min_val_loss_index = training_progress['Validation Loss'].idxmin()
    
    # Get the index of the last epoch
    last_epoch_index = training_progress.index[-1]
    
    # Check if the difference between indices exceeds the threshold
    if (last_epoch_index - min_val_loss_index) > threshold:
        print('Early stopping applied')
        return True
    return False