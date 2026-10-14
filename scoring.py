from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



################# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])

FEATURES = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET = 'exited'


def load_model(path):
    """Load a pickled model from path."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_test_data():
    """Load test_data_path/testdata.csv."""
    return pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))


################# Function for model scoring
def score_model(model, dataframe, score_path=None):
    """Calculate and return the supplied model's F1 score on the supplied DataFrame."""
    x = dataframe[FEATURES]
    y = dataframe[TARGET]

    predictions = model.predict(x)
    f1_score = metrics.f1_score(y, predictions)

    if score_path is not None:
        os.makedirs(os.path.dirname(score_path) or '.', exist_ok=True)
        with open(score_path, 'w') as f:
            f.write(str(f1_score))

    return f1_score


if __name__ == '__main__':
    trained_model = load_model(os.path.join(model_path, 'trainedmodel.pkl'))
    score = score_model(trained_model, load_test_data(),
                        score_path=os.path.join(model_path, 'latestscore.txt'))
    print(f"F1 score: {score}")

