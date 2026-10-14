import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

from diagnostics import model_predictions


matplotlib.use('Agg')

############### Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

TARGET = 'exited'


############## Function for reporting
def score_model(output_filename='confusionmatrix.png'):
    """Plot the deployed model's confusion matrix on the test data."""
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_true = test_data[TARGET]
    y_pred = model_predictions(test_data)

    labels = [0, 1]
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix (Test Data)')
    fig.tight_layout()

    os.makedirs(output_model_path, exist_ok=True)
    fig.savefig(os.path.join(output_model_path, output_filename))
    plt.close(fig)

    return matrix


if __name__ == '__main__':
    score_model(*sys.argv[1:2])
