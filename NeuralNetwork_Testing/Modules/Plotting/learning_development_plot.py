import matplotlib.pyplot as plt
import pandas as pd
import os

def learning_development_plot(training:pd.DataFrame, Normalizer=None, saving_path=None, show=False):
    if Normalizer is not None:
        raise NotImplementedError('Normalizer is not implemented yet.')
    plt.figure(figsize=(15, 5))
    plt.plot(training['Loss'], label='Loss')
    plt.plot(training['Validation Loss'], label='Validation Loss')
    plt.legend()
    if saving_path is not None:
        plt.savefig(os.path.join(saving_path,'Training_Development.svg'))
    if show:
        plt.show()

if __name__ == '__main__':
    training = pd.read_csv('/home/u108-n256/PalmProject/CrossValidation/Model_11/Fold_1_TrainingProgress.csv')
    learning_development_plot(training, saving_path='/home/u108-n256/PalmProject/CrossValidation/Model_11', show=True)