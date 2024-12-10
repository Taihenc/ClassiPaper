import os
import json
import pandas as pd

def GetCleanedData():
    # Define the path to the cleaned data file
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'cleaned_data.json')
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the JSON data into a DataFrame
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    df = pd.DataFrame(data)
    return df
print(GetCleanedData().head())