import shap
import numpy as np
import random
import torch
from Modules.Datasets import BaseDatasetPalm
from Modules.Shapley import create_kmeans_samples

def shapley_test(X, model, random_seed, size_background_data:int=100, nsamples:int=5):

    # 3D --> 2D for kmeans
    X = X.reshape(-1, X.shape[1]*X.shape[2])

    random.seed(random_seed)
    background = create_kmeans_samples(X,clusters=size_background_data, random_state=0)
    ksamples = np.array(random.choices(X,k=nsamples))


    print('BB-Function is intialized')
    def black_box_function(X: np.array) -> np.array:
        """
        Black Box Function for the SHAP explainer.

        This function is designed to interface with SHAP (SHapley Additive exPlanations) explainers. It takes input data in the form of a 2D-NumPy (since the shap.KernelExplainer only takes this input) array,
        processes it to match the expected 3D input shape (sample, time, feature) of the given model, and then performs a forward pass through the model to obtain prediction.

        Parameters:
        -----------
        X : np.array
            The input data array. It is expected to be a 3D array where the first dimension is the batch size.
            The function reshapes it to match the expected input dimensions for the model.
        model : torch.nn.Module
            The PyTorch model used for making prediction. The model should already be trained and capable of running on the available device (CPU or GPU).

        Returns:
        --------
        np.array
            The prediction made by the model. The output array has the same shape as the input array `X`.
        np.array
            The samples created from kmeans clustering and used for 

        Steps:
        ------
        1. Determine the device to run the model on (GPU if available, otherwise CPU).
        2. Move the model to the selected device.
        3. Reshape the input array `X` to the shape expected by the model (batch size, 40, 190).
        4. Convert the input array `X` to a PyTorch tensor and move it to the selected device.
        5. Perform a forward pass through the model to obtain prediction.
        6. Move the prediction back to the CPU, detach them from the computational graph, convert them to a NumPy array, and reshape them to match the input shape `shape_2d`.

        Example:
        --------
        >>> import numpy as np
        >>> from some_model_library import YourModelClass
        >>> model = YourModelClass()
        >>> model.load_state_dict(torch.load("your_model_path.pt"))
        >>> X = np.random.rand(10, 40 * 190)
        >>> prediction = bb_function(X, model)
        >>> print(prediction.shape)
        (10, 40, 190)
        """
        print(f'Input shape: {X.shape}\n')
        predictions_output = np.zeros((X.shape[0]))

        # Determine the device to use (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the selected device
        model.to(device)
        model.eval()

        # Reshape the input array to match the expected input shape of the model (batch size, 40, 190)
        X = X.reshape(-1,190,40)

        for i, sample in enumerate(X):
            # Convert the input array to a PyTorch tensor and move it to the selected device
            sample = torch.tensor(sample, dtype=torch.float32).to(device)

            # Perform a forward pass through the model to obtain prediction
            prediction = model(sample.unsqueeze(0))
            
            predictions_output[i] = prediction.squeeze().detach().cpu().numpy()

        print(f'Output shape: {predictions_output.shape}\n')
        # Move the prediction back to the CPU, detach them from the computational graph, convert to a NumPy array, and reshape to match the input shape `shape_2d`
        return predictions_output + np.where(np.zeros_like(predictions_output) == 0, 10e-10, 10e-10)

    # Create the SHAP explainer
    print('SHAP explainer is intialized\n')
    explainer = shap.KernelExplainer(black_box_function, background)

    #Calculate the SHAP values from X --> shapley values has same number of features (observation period x features)
    print('SHAP values are calculated\n')
    shapley_values = explainer(ksamples)
    print(f'SHAP values are calculated with shape {shapley_values.shape}\n')
    return (shapley_values, np.array(ksamples))

if __name__ == 'main':
    pass