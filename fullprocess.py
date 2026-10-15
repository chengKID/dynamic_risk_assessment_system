import os
import sys
import ast
import json
import time
import signal
import subprocess
from datetime import datetime

import pandas as pd
import requests

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting


with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
test_data_path = config['test_data_path']
output_model_path = config['output_model_path']
prod_deployment_path = config['prod_deployment_path']

API_URL = 'http://127.0.0.1:8000'


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def read_ingested_files(path):
    """Read an ingestedfiles.txt record (a Python list) into a set of file names."""
    with open(path, 'r') as f:
        content = f.read().strip()
    return set(ast.literal_eval(content)) if content else set()


def api_is_up():
    try:
        requests.get(f"{API_URL}/summarystats", timeout=5)
        return True
    except requests.exceptions.RequestException:
        return False


def run_apicalls(output_filename):
    """Run apicalls.py against the API, starting app.py temporarily if needed."""
    server = None
    if not api_is_up():
        log("API not running - starting app.py")
        # New session so the Flask debug reloader's child process is stopped too
        server = subprocess.Popen([sys.executable, 'app.py'], start_new_session=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            if api_is_up():
                break
            time.sleep(1)
        else:
            os.killpg(server.pid, signal.SIGTERM)
            raise RuntimeError("app.py did not start within 60 seconds")
    try:
        subprocess.run([sys.executable, 'apicalls.py', output_filename], check=True)
    finally:
        if server is not None:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait()


def main():
    deployed_files = read_ingested_files(os.path.join(prod_deployment_path, 'ingestedfiles.txt'))
    source_files = {name for name in os.listdir(input_folder_path) if name.lower().endswith('.csv')}
    new_files = sorted(source_files - deployed_files)

    if not new_files:
        log("No new data found - stopping.")
        return
    log(f"New data found: {new_files} - ingesting.")
    subprocess.run([sys.executable, 'ingestion.py'], check=True)

    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
        baseline_score = float(f.read().strip())
    deployed_model = scoring.load_model(os.path.join(prod_deployment_path, 'trainedmodel.pkl'))
    new_data = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    new_score = scoring.score_model(deployed_model, new_data)  # no score_path: baseline untouched
    log(f"Deployed baseline F1 = {baseline_score:.4f}, F1 on new data = {new_score:.4f}")

    if new_score < baseline_score:
        log("Model drift detected - retraining and evaluating candidate.")
        subprocess.run([sys.executable, 'training.py'], check=True)
        new_model = scoring.load_model(os.path.join(output_model_path, 'trainedmodel.pkl'))
        test_data = scoring.load_test_data()
        production_test_score = scoring.score_model(deployed_model, test_data)
        test_score = scoring.score_model(
            new_model, test_data,
            score_path=os.path.join(output_model_path, 'latestscore.txt'),
        )
        log(f"Retrained model F1 on test data = {test_score:.4f}")
        log(f"Production model F1 on the same test data = {production_test_score:.4f}")
        if test_score > production_test_score:
            subprocess.run([sys.executable, 'deployment.py'], check=True)
            log(f"Deployed improved model to {prod_deployment_path}.")
        else:
            log("Candidate did not improve test F1 - keeping production unchanged.")
    else:
        log("No model drift - keeping the deployed model and baseline score.")

    reporting.score_model('confusionmatrix2.png')
    log(f"Saved {os.path.join(output_model_path, 'confusionmatrix2.png')}")
    run_apicalls('apireturns2.txt')
    log(f"Saved {os.path.join(output_model_path, 'apireturns2.txt')}")


if __name__ == '__main__':
    main()