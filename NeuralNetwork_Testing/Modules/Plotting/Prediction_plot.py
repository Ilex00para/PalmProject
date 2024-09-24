import matplotlib.pyplot as plt
import pandas as pd
import os

def Prediction_plot(predictions:pd.DataFrame, title='Title',Normalizer=None, saving_path=None, show=False):
    if Normalizer is not None:
        raise NotImplementedError('Normalizer is not implemented yet.')
    plt.figure(figsize=(15, 5))
    plt.plot(predictions['Predicted'], label='Predictions')
    plt.plot(predictions['Actual'], label='True values')
    plt.legend()
    plt.xlabel('Actual Flower Number (#)')
    plt.ylabel('Predicted Flower Number (#)')
    plt.title(title)
    if saving_path is not None:
        plt.savefig(os.path.join(saving_path,'Predictions.svg'))
    if show:
        plt.show()