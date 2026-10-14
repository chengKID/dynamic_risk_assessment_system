import os
import json
import sys

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


def main(output_filename='apireturns.txt'):
    responses = {
        'prediction': call('POST', '/prediction', json={'filepath': 'testdata/testdata.csv'}),
        'scoring': call('GET', '/scoring'),
        'summarystats': call('GET', '/summarystats'),
        'diagnostics': call('GET', '/diagnostics'),
    }

    os.makedirs(output_model_path, exist_ok=True)
    with open(os.path.join(output_model_path, output_filename), 'w') as f:
        json.dump(responses, f, indent=4)


if __name__ == '__main__':
    main(*sys.argv[1:2])
