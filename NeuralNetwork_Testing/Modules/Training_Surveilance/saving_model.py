import os
import torch

def create_folder(directory, folder_name):
        # Check if the directory exists, create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    if folder_name is None:
        folder_path = directory
    else:
        counter = 0
        while os.path.exists(os.path.join(directory, folder_name)):
            counter += 1
            folder_name = f"{folder_name.split('_')[0]}_{counter}"
        folder_path = os.path.join(directory, folder_name)


    # Check if the folder exists, create it if it doesn't
    if folder_name and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return folder_path
    

def saving_model(model, training_development, predictions, directory='./NeuralNetwork_Testing/Saved_Objects', folder_name='Model'):
    folder_path = create_folder(directory, folder_name)
    torch.save(model.state_dict(), os.path.join(folder_path, 'Model.pt'))
    training_development.to_csv(os.path.join(folder_path,'training_progress.csv'), index=False)
    predictions.to_csv(os.path.join(folder_path,'predictions.csv'), index=False)
    print(f'Model saved at {folder_path}')



