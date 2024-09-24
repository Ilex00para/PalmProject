import torch

from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from Modules import CNN_MNV2, BaseDatasetPalm, full_training, saving_model, cyclicLR_step_size, Prediction_plot, Transformer_Encoder

#Hyperparameters
epochs = 100
batch_size = 64 #64


training_dataset = BaseDatasetPalm('female', site=['SMSE','PR'],time_windows=[0,39,49])
testing_dataset = BaseDatasetPalm('female', site=['SMSE'],train=False, time_windows=[0,39,49])

trainining_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

#calculates the step_size for the cyclicLR scheduler
step_size = cyclicLR_step_size(5,(training_dataset.X.shape[0]),batch_size)

#chooses GPU if available to run the training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#import the model architecture
model = CNN_MNV2()
model.to(device)

#optimizer and loss function are defined
optimizer = torch.optim.Adam(model.parameters(), lr=10e-8, weight_decay=1e-2)
loss_function = torch.nn.MSELoss()

#cycles the learning rate between 'base_lr' and max_lr' values, every 'step_size' iterations the the 'max_lr' value is halfed
lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-6, step_size_up=step_size, mode='triangular2')


#trains the model and saves the best model, training progress and predictions
#early stopping is enabled
model,best_model,training_progress, predictions = full_training(model, 
                                                    trainining_dataloader, 
                                                    testing_dataloader, 
                                                    optimizer=optimizer, 
                                                    loss_function=loss_function, 
                                                    epochs=epochs, 
                                                    device=device,
                                                    lr_scheduler=lr_scheduler,
                                                    verbose=True,
                                                    gradients=True,
                                                    early_stopping=True)

torch.save(model, '/home/u108-n256/PalmProject/CrossValidation/OPTIM/Optimized_CNN.pt')

torch.save(best_model, '/home/u108-n256/PalmProject/CrossValidation/OPTIM/Optimized_CNN_best.pt')

training_progress.to_csv('/home/u108-n256/PalmProject/CrossValidation/OPTIM/Optimized_CNN_TrainingProgress.csv', index=False)

predictions.to_csv('/home/u108-n256/PalmProject/CrossValidation/OPTIM/Optimized_CNN_Predictions.csv', index=False)
