import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def Confusion_Matrix_Regression(y_true, y_pred):
    """
    Creates a confusion matrix for regression data by binning the continuous values into discrete categories.
    
    Parameters:
    y_true (array-like): True values
    y_pred (array-like): Predicted values
    bins (int): Number of bins to categorize the continuous values
    
    Returns:
    confusion_matrix (array): The confusion matrix
    """
    # Create bin edges for the true and predicted values
    bin_edges = np.linspace(min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred)),1)
    print(bin_edges)
    # Bin the true and predicted values
    y_true_binned = np.digitize(y_true, bins=bin_edges) - 1
    y_pred_binned = np.digitize(y_pred, bins=bin_edges) - 1

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_true_binned, y_pred_binned)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Bins')
    plt.ylabel('True Bins')
    plt.title('Confusion Matrix for Regression')
    plt.show()

    return conf_matrix





if __name__ == '__main__':
    # Example usage
    np.random.seed(0)
    y_true = np.random.normal(0, 1, 100)
    y_pred = y_true + np.random.normal(0, 0.5, 100)

    conf_matrix = Confusion_Matrix_Regression(y_true, y_pred)
    print(conf_matrix)
