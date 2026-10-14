import os
import json

import requests


# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])


def call(method, endpoint, **kwargs):
    """Call an endpoint, fail loudly on a non-200 status, return the JSON body."""
    response = requests.request(method, f"{URL}{endpoint}", timeout=600, **kwargs)
    response.raise_for_status()
    return response.json()


# Call each API endpoint and store the responses
response1 = call('POST', '/prediction', json={'filepath': 'testdata/testdata.csv'})
response2 = call('GET', '/scoring')
response3 = call('GET', '/summarystats')
response4 = call('GET', '/diagnostics')

# Combine all API responses
responses = {
    'prediction': response1,
    'scoring': response2,
    'summarystats': response3,
    'diagnostics': response4,
}

# Write the responses to your workspace
os.makedirs(output_model_path, exist_ok=True)
with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as f:
    json.dump(responses, f, indent=4)
