import json
import os
import pandas as pd
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv


load_dotenv()
# Define the Paper dataclass
@dataclass
class Paper:
    title: str
    keywords: str
    abstract: str
    subject_area: str

# Function to load the data into a pandas DataFrame
def GetCleanedData() -> pd.DataFrame:
    # Get the environment variables
    base_cleaned_dir = os.getenv("BASE_CLEANED_DIR", "").strip().strip('"')
    base_cleaned_file = os.getenv("BASE_CLEANED_FILE", "").strip().strip('"')
    
    # Construct the full file path
    project_root = os.path.abspath((os.path.join(os.path.dirname(__file__), '../..')))
    file_path = os.path.join(project_root, base_cleaned_dir, base_cleaned_file)
    
    # Read the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create a list of Paper objects from the loaded data
    papers = [Paper(
                title=paper['clean_title'],
                keywords=paper['clean_keywords'],
                abstract=paper['clean_abstract'],
                subject_area=paper['clean_subject_area']
              ) for paper in data]

    # Convert the list of Paper objects to a list of dictionaries
    papers_dict = [paper.__dict__ for paper in papers]

    # Create and return the DataFrame
    return pd.DataFrame(papers_dict)

df = GetCleanedData()
df.head()
df.info()