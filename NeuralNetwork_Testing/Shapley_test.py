# %%
import shap
import shap.maskers
import torch
import numpy as np
import matplotlib.pyplot as plt
import os 

import NN_Architectures.exclude_Flowers_Bunchload as arch_ex
import NN_Architectures.full_data as arch_full
import NN_Architectures.exclude_Flowers as arch_ex_bl
from Functions.DataSet import PhenologicalDatasetTransposedMA_minus, PhenologicalDatasetTransposedMA, PhenologicalDatasetTransposedMA_minus_flowers
from sklearn.cluster import KMeans
np.random.seed(42) # 42

# %%


# %%
# Load the models ...
#...Architecture
for flower in ['male']:
    for data_id, model_ in enumerate(zip([arch_ex.Flowering_TransformerCNN.TransformerCNN(),
                  arch_ex.Flowering_CNN.CNN(),
                  arch_ex.Flowering_Transformer.Transformer(),
                  arch_ex_bl.Flowering_TransformerCNN.TransformerCNN(),
                  arch_ex_bl.Flowering_CNN.CNN(),
                  arch_ex_bl.Flowering_Transformer.Transformer(),
                  arch_full.Flowering_TransformerCNN.TransformerCNN(),
                  arch_full.Flowering_CNN.CNN(),
                  arch_full.Flowering_Transformer.Transformer()],
                  [None,#'/home/u108-n256/PalmProject/NeuralNetwork_Testing/Saved_Objects/BEST_MODELS/all/Model_CNN_Transformer_male_nofbl/MODEL_V4.pt',
                   None,
                   None,
                   None,#'/home/u108-n256/PalmProject/NeuralNetwork_Testing/Saved_Objects/BEST_MODELS/all/Model_CNN_Transformer_male_nof/MODEL_V1.pt',
                   None,
                   None,
                   '/home/u108-n256/PalmProject/NeuralNetwork_Testing/Saved_Objects/BEST_MODELS/all/Model_CNN_Transformer_male_full/MODEL_V7.pt',
                   None,
                   None  
                  ])):
        try:
            if model_[0] is None:
                continue
            
            model = model_[0]
            
            
            #...Parameters
            model.load_state_dict(torch.load(model_[1]))
            print('Model loaded')

            input_data_dir = '/home/u108-n256/PalmProject/NeuralNetwork_Testing/NN_Inputs'
            if data_id < 3:
                X = PhenologicalDatasetTransposedMA_minus(train=False, flower=flower, path=input_data_dir)
            elif data_id < 6:
                X = PhenologicalDatasetTransposedMA_minus_flowers(train=False, flower=flower, path=input_data_dir)
            else:
                X = PhenologicalDatasetTransposedMA(train=False, flower=flower, path=input_data_dir)

            #Save the shape of the data (samples, periods of observation, features)
            #Features shiuld be the only one changing
            X = X.data
            x_shape = X.shape

            # Use KMeans to create a background dataset which represents the whole dataset
            kmeans = KMeans(n_clusters=10, random_state=0).fit(X.reshape(-1, x_shape[1]*x_shape[2]))
            background = kmeans.cluster_centers_

            #draw random samples from the data
            random_ids = np.random.choice(X.shape[0], 100, replace=False)



            def f(x: np.array) -> np.array:
                '''
                The "Black Box Function" which is used to calculate the SHAP values.
                It first reshapes x back to the original shape.
                After the reshaped input data are used to calculate the output of the model.


                args:
                x : np.array
                    The input data for the model/ network

                returns:
                output : np.array
                    The output of the model/ network
                
                
                '''
                print(f'Input of the Model {x.shape}')
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f'Device: {device}')

                output = torch.zeros((x.shape[0])).to(device)

                # Reshape the input to the correct shape

                model.to(device)
                model.eval()
                x = torch.tensor(x, dtype=torch.float32).to(device)

                for sample in range(x.shape[0]):
                    sample_data = x[sample,:].reshape(-1, x_shape[1], x_shape[2])
                    input = torch.tensor(sample_data, dtype=torch.float32).clone().detach()
                    #input = torch.cat((input, input, input), dim=0)
                    
                    output[sample] = torch.mean(model(input).clone().detach())
                
                torch.cuda.empty_cache()
                
                output.reshape(-1)

                print(f'Output of the Model {output.shape}')
                return output.cpu().detach().numpy()


            # %%
            """The number of pertubated combinations is 2^35, which is too large to compute.
                nsamples : "auto" or int
                    Number of times to re-evaluate the model when explaining each prediction. More samples
                    lead to lower variance estimates of the SHAP values. The "auto" setting uses
                    `nsamples = 2 * X.shape[1] + 2048`."""

            # Create the SHAP explainer
            explainer = shap.KernelExplainer(f, background)

            #Create the input data for the explainer with the previously drawn samples
            X_shapley =X[random_ids,:,:].reshape(-1, x_shape[1]*x_shape[2])

            #Calculate the SHAP values from X --> shapley values has same number of features (observation period x features)
            shapley_values = explainer(X_shapley)



            # %%
            print(shapley_values.values)
            print(shapley_values.values.shape)
            print(explainer.expected_value)
            print('SHAP values calculated')

            # Determine the project root directory dynamically
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

            # Define the relative path to the target directory


            cwd = os.path.join(project_root,'NeuralNetwork_Testing', 'Saved_Objects', 'Shapley-Values')
            path  = 'SHAP_values.npy'
            counter = 0
            while os.path.exists(os.path.join(cwd, path)):
                counter += 1
                path = path.split('.')[0] + f'{counter}.npy'

            try:
                np.save(os.path.join(cwd, path), shapley_values.values)

            except FileNotFoundError:
                os.makedirs(cwd)
                np.save(os.path.join(cwd, path), shapley_values.values)

            print(f'SHAP values saved at {os.path.join(cwd, path)}')
        except Exception as e:
            print(f'Error: {e}')
            continue

