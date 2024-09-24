import skorch
import torch
import pickle
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
from Architectures import CNN_MNV2, CNN_v0, EfficientNet, Transformer_Encoder

from torch.utils.data import DataLoader
from Datasets import CrossValidationDataset, BaseDatasetPalm



dataset = BaseDatasetPalm(train=True, flower='male')
train_X, train_y = dataset.X, dataset.y

train_X = torch.tensor(train_X.reshape(train_X.shape[0], -1), dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)


print(train_X.shape, train_y.shape)


model2 =  NeuralNetRegressor(module=Transformer_Encoder(n_heads=2,n_layers=2),
                            batch_size=64,
                            max_epochs = 50,
                            lr=1e-6,
                            module__n_layers=2,
                            module__n_heads = 2,
                            optimizer__weight_decay=1e-4,
                            device='cuda')

model3 = NeuralNetRegressor(module=Transformer_Encoder(n_heads=2,n_layers=2),
                            batch_size=64,
                            max_epochs = 50,
                            lr=1e-6,
                            optimizer = torch.optim.Adam,
                            optimizer__weight_decay=1e-4,
                            device='cuda')

para_grid = {
    'module__n_layers' : [2,4,6,8],
    'module__n_heads' : [2, 5, 10],
    'batch_size' : [2,4,8,16],
    'lr': [1e-4, 1e-5, 1e-6]
}

grid = GridSearchCV(model2, param_grid=para_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1, error_score='raise')

grid_result = grid.fit(train_X, train_y)


try:
    with open('./grid_result_encoder_male.pkl', 'wb') as file:
        pickle.dump(grid_result, file)
    with open('./grid_result_encoder_202407241636.txt', 'w') as file:
        file.write(f'{grid_result.best_params_, grid_result.best_score_, grid_result.cv_results_}')
except:
    raise Exception('Error saving grid search results')

"""
grid = GridSearchCV(model3, param_grid=para_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1, error_score='raise')

grid_result = grid.fit(train_X, train_y)


try:
    with open('./grid_result_encoder_female.pkl', 'wb') as file:
        pickle.dump(grid_result, file)
    with open('./grid_result_encoder_female.txt', 'w') as file:
        file.write(f'{grid_result.best_params_, grid_result.best_score_, grid_result.cv_results_}')
except:
    raise Exception('Error saving grid search results')
"""