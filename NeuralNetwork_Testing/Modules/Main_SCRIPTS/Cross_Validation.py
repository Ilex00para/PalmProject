import os

import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader

from Modules.Datasets import BaseDatasetPalm, CrossValidationDataset
from .Full_Training import full_training
from Modules.Training_Surveilance import get_best_metric, saving_to_folder
from Modules.Utils import cyclicLR_step_size
from Modules.Plotting import Prediction_plot


def cross_validation(dataset: BaseDatasetPalm, model_class=nn.Module, kfolds=5, epochs=100, batch_size=64, verbose=True, **kwargs):
    """
    Performs k-times a cross-validation on a dataset (BaseDatasetPalm) using a given model returning the average RMSE of all folds. 

    Args:
        dataset (BaseDatasetPalm): The dataset used for training and testing. Needs to be instantiated before with given args mask, flower and time window.
        model_class (torch.nn.Module): The class of the model to be trained.
        kfolds (int): Number of folds for cross-validation. Defaults to 5.
        epochs (int): Number of epochs to train the model. Defaults to 100.
        verbose (bool): If True, prints progress information. Defaults to True.
        **kwargs: Additional arguments.
            - folder_name (str): Name of the folder to save models and training progress.
            - save_each_model (bool): If True, saves each model after training.
    
    Returns:
        tuple: A tuple containing the mean RMSE across all folds and a DataFrame with RMSE for each fold.
    """
    #Creates the df for storing the RMSEs for each fold
    scores = pd.DataFrame(columns=['Fold', 'RMSE'])
    #Creates the folder to store the models and training progress from path+folder_name
    folder_name = 'Model_0' if kwargs.get('folder_name') is None else kwargs.get('folder_name')
    path = '/home/u108-n256/PalmProject/CrossValidation'

    if os.path.exists(os.path.join(path, folder_name)) == False:
        os.mkdir(os.path.join(path, folder_name))

    # Cross-validation loop
    # create the kfold object which returns the indices for the training and testing sets
    kfold = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    
    for fold, ids in enumerate(kfold.split(dataset.X)):
        # Preparation of data
        train_ids, test_ids = ids[0], ids[1]

        #Split the data into training and testing
        train_X, test_X = dataset.X[train_ids], dataset.X[test_ids]
        train_y, test_y = dataset.y[train_ids], dataset.y[test_ids]

        #Create DataLoaders form the training and testing data
        training_dataloader = DataLoader(CrossValidationDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
        testing_dataloader = DataLoader(CrossValidationDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

        # Initilization Model, Loss, and Optimizer 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
        loss_function = torch.nn.MSELoss()
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-7, max_lr=1e-5, step_size_up=cyclicLR_step_size(5,train_X.shape[0],batch_size=batch_size), mode='triangular2')

        # Training of the model
        _, best_model, training_progress, predictions = full_training(model, training_dataloader, testing_dataloader, optimizer, loss_function, 
                                                            epochs=epochs, 
                                                            device=device, 
                                                            lr_scheduler=lr_scheduler, 
                                                            track_metrics=True, 
                                                            verbose=True,
                                                            early_stopping=True)

        # Storing best RMSEs
        scores = pd.concat([scores, pd.Series({'Fold': int(fold) + 1, 'RMSE': training_progress['RMSE'].iloc[get_best_metric(training_progress).idx]}).to_frame().T], ignore_index=True)

        # Print progress
        if verbose:
            print(f'Fold {int(fold) + 1} completed. RMSE: {scores["RMSE"].iloc[-1]}')

        # Save each model per fold if specified
        if kwargs.get('save_each_model'):
            torch.save(best_model, os.path.join(os.getcwd(), 'CrossValidation', folder_name, f'Fold_{int(fold) + 1}_Model.pt'))
            saving_to_folder(training_progress, directory=os.path.join(os.getcwd(), 'CrossValidation'), folder_name=folder_name, file_name=f'Fold_{int(fold) + 1}_TrainingProgress')
            saving_to_folder(predictions, directory=os.path.join(os.getcwd(), 'CrossValidation'), folder_name=folder_name, file_name=f'Fold_{int(fold) + 1}_Predictions')
            if epochs > 0:
                Prediction_plot(predictions.iloc[get_best_metric(training_progress).idx], saving_path=os.path.join(os.getcwd(), 'CrossValidation', folder_name), show=False)

    # Save the best scores in each folder
    saving_to_folder(scores, directory=os.path.join(os.getcwd(), 'CrossValidation'), folder_name=folder_name, file_name='Scores') 
    return scores['RMSE'].mean(), scores