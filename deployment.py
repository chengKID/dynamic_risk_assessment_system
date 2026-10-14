from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


################## Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])


#################### Function for deployment
def store_model_into_pickle(model=None):
    """Copy trainedmodel.pkl, latestscore.txt and ingestedfiles.txt to production."""
    model_file = os.path.join(model_path, model or 'trainedmodel.pkl')
    score_file = os.path.join(model_path, 'latestscore.txt')
    ingested_file = os.path.join(dataset_csv_path, 'ingestedfiles.txt')

    os.makedirs(prod_deployment_path, exist_ok=True)
    for path in (model_file, score_file, ingested_file):
        shutil.copy2(path, prod_deployment_path)


if __name__ == '__main__':
    store_model_into_pickle()
