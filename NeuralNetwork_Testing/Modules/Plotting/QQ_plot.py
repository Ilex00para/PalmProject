import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def QQ_plot(predictions:pd.DataFrame, title='Title', Normalizer=None, saving_path=None, show=False, point_color='blue', alpha=1):
    if Normalizer is not None:
        raise NotImplementedError('Normalizer is not implemented yet.')
    group_index = np.arange(len(predictions)) // 18
    grouped_df = predictions.groupby(group_index).sum().reset_index(drop=True)
    print(grouped_df)
    plt.figure(figsize=(10, 10))
    plt.scatter(grouped_df['Actual'], grouped_df['Predicted'],c=point_color, alpha=alpha)
    plt.xlabel('Real Flower Number')
    plt.ylabel('Predicted Flower Number')
    plt.xlim(-0.1,grouped_df['Actual'].values.max()+1)
    plt.ylim(-0.1,grouped_df['Actual'].values.max()+1)
    plt.xticks(np.arange(0,grouped_df['Actual'].values.max()+1,2),np.arange(0,grouped_df['Actual'].values.max()+1,2,dtype=int))
    plt.yticks(np.arange(0,grouped_df['Actual'].values.max()+1,2),np.arange(0,grouped_df['Actual'].values.max()+1,2,dtype=int))
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='red', linewidth=1)
    plt.title(title)
    if saving_path is not None:
        plt.savefig(os.path.join(saving_path,'QQ_plot.svg'))
    if show:
        plt.show()
