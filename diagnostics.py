import os
import re
import sys
import json
import pickle
import shutil
import tempfile
import subprocess
import timeit

import pandas as pd


################## Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

FEATURES = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']


################## Function to get model predictions
def model_predictions(dataframe):
    """Predict with the deployed model; returns one prediction per input row."""
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(dataframe[FEATURES]).tolist()
    return predictions

################## Function to get summary statistics
def dataframe_summary():
    """Mean, median and standard deviation for every numeric column of finaldata.csv."""
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    numerics = data.select_dtypes(include='number')

    summary = [
        {
            'column': col,
            'mean': float(numerics[col].mean()),
            'median': float(numerics[col].median()),
            'std': float(numerics[col].std()),
            'mode': float(numerics[col].mode().iloc[0]),
        }
        for col in numerics.columns
    ]
    return summary


################## Function to check missing data
def missing_data():
    """Percent of NA values per column of finaldata.csv (all columns, in column order)."""
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    return (data.isna().sum() / len(data) * 100).tolist()


##################Function to get timings
def execution_time():
    timings = []

    for script in ('ingestion.py', 'training.py'):
        start = timeit.default_timer()
        subprocess.run([sys.executable, script], check=True)
        timings.append(timeit.default_timer() - start)
    return timings

################## Function to check dependencies
def _pip(*args):
    """Run a pip command with the current interpreter and return stdout."""
    result = subprocess.run([sys.executable, '-m', 'pip', *args],
                            capture_output=True, text=True)
    return result.stdout


def _normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def outdated_packages_list(requirements_file='requirements.txt'):
    """Return a table of package, current (pinned) version and latest version."""
    requirements = {}
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if '==' not in line:
                continue
            package, version = (part.strip() for part in line.split('==', 1))
            requirements[package] = version

    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
        check=True,
        capture_output=True,
        text=True,
    )
    latest_versions = {
        _normalize(item['name']): item['latest_version']
        for item in json.loads(result.stdout or '[]')
    }

    table = pd.DataFrame(
        [
            {
                'package': package,
                'current_version': current_version,
                'latest_version': latest_versions.get(_normalize(package), current_version),
            }
            for package, current_version in requirements.items()
        ],
        columns=['package', 'current_version', 'latest_version'],
    )
    print(table.to_string(index=False))
    return table


if __name__ == '__main__':
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    print('Predictions:', model_predictions(test_data))
    print('Summary statistics:', dataframe_summary())
    print('Missing data (%):', missing_data())
    print('Execution time (s) [ingestion, training]:', execution_time())
    outdated_packages_list()





    
