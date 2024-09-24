import numpy as np
import torch


"BASIC EXAMPLE OF A BLACK BOX FUNCTION FOR SHAP EXPLAINER"

def black_box_function(X: np.array) -> np.array:
    """
    Black Box Function for the SHAP explainer.

    This function is designed to interface with SHAP (SHapley Additive exPlanations) explainers. It takes input data in the form of a NumPy array,
    processes it to match the expected input shape of the given model, and then performs a forward pass through the model to obtain predictions.

    Parameters:
    -----------
    X : np.array
        The input data array. It is expected to be a 3D array where the first dimension is the batch size.
        The function reshapes it to match the expected input dimensions for the model.
    model : torch.nn.Module
        The PyTorch model used for making predictions. The model should already be trained and capable of running on the available device (CPU or GPU).

    Returns:
    --------
    np.array
        The predictions made by the model. The output array has the same shape as the input array `X`.

    Steps:
    ------
    1. Determine the device to run the model on (GPU if available, otherwise CPU).
    2. Move the model to the selected device.
    3. Reshape the input array `X` to the shape expected by the model (batch size, 40, 190).
    4. Convert the input array `X` to a PyTorch tensor and move it to the selected device.
    5. Perform a forward pass through the model to obtain predictions.
    6. Move the predictions back to the CPU, detach them from the computational graph, convert them to a NumPy array, and reshape them to match the input shape `s`.

    Example:
    --------
    >>> import numpy as np
    >>> from some_model_library import YourModelClass
    >>> model = YourModelClass()
    >>> model.load_state_dict(torch.load("your_model_path.pt"))
    >>> X = np.random.rand(10, 40 * 190)
    >>> predictions = bb_function(X, model)
    >>> print(predictions.shape)
    (10, 40, 190)
    """

    # Get the shape of the input array
    s = X.shape

    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the selected device
    model.to(device)

    # Reshape the input array to match the expected input shape of the model (batch size, 40, 190)
    X = X.reshape(-1, 40, 190)

    # Convert the input array to a PyTorch tensor and move it to the selected device
    X = torch.tensor(X, dtype=torch.float32).to(device)

    # Perform a forward pass through the model to obtain predictions
    predictions = model(X)

    # Move the predictions back to the CPU, detach them from the computational graph, convert to a NumPy array, and reshape to match the input shape `s`
    return predictions.cpu().detach().numpy().reshape(s)
