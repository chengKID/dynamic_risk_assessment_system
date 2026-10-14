import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




############# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



############# Function for data ingestion
def merge_multiple_dataframe():
    """Combine all CSV files in input_folder_path into one de-duplicated dataset."""
    filenames = sorted(
        name for name in os.listdir(input_folder_path)
        if name.lower().endswith('.csv')
    )
    dataframes = [pd.read_csv(os.path.join(input_folder_path, name)) for name in filenames]

    final_df = pd.DataFrame()
    if dataframes:
        final_df = pd.concat(dataframes, ignore_index=True).drop_duplicates().reset_index(drop=True)

    os.makedirs(output_folder_path, exist_ok=True)
    final_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.write(str(filenames))

    return final_df


if __name__ == '__main__':
    merge_multiple_dataframe()
