from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

import diagnostics
import scoring


###################### Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


####################### Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """Predict with the deployed model on the dataset at the supplied file location."""
    if request.method == 'OPTIONS':
        return jsonify(), 200

    payload = request.get_json(silent=True) or {}
    filepath = payload.get('filepath') or request.form.get('filepath') or request.args.get('filepath')
    if not filepath:
        return jsonify({'error': "Provide the dataset location as 'filepath'."}), 400
    if not os.path.isfile(filepath):
        return jsonify({'error': f"File not found: {filepath}"}), 400

    dataframe = pd.read_csv(filepath)
    predictions = diagnostics.model_predictions(dataframe)
    return jsonify({'predictions': predictions}), 200


####################### Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring_endpoint():
    """F1 score of the deployed model on test_data_path/testdata.csv."""
    if request.method == 'OPTIONS':
        return jsonify(), 200

    model = scoring.load_model(os.path.join(prod_deployment_path, 'trainedmodel.pkl'))
    dataframe = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    score = scoring.score_model(model, dataframe)
    return jsonify({'f1_score': score}), 200


####################### Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats_endpoint():
    """Means, medians and standard deviations of the numeric columns."""
    if request.method == 'OPTIONS':
        return jsonify(), 200

    summary_statistics = diagnostics.dataframe_summary()
    return jsonify({'summary_statistics': summary_statistics}), 200


####################### Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics_endpoint():
    """Timings, missing-value percentages and the dependency version table."""
    if request.method == 'OPTIONS':
        return jsonify(), 200
    
    return jsonify({
        'execution_time_seconds': diagnostics.execution_time(),
        'missing_data_percent': diagnostics.missing_data(),
        'dependencies': diagnostics.outdated_packages_list().to_dict(orient='records'),
    }), 200

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
