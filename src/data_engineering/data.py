import os
import json
from typing import List
from dataclasses import dataclass

# Define the Paper type
@dataclass
class Paper:
    title: str
    keywords: List[str]
    abstract: str

# Function to load the cleaned papers
def get_cleaned_paper(file_path: str) -> Paper:
    """Loads a cleaned paper from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Return a Paper object using cleaned data
            return Paper(
                title=data.get('clean_title', ''),
                keywords=data.get('clean_keywords', '').split(),  # Convert space-separated string to list
                abstract=data.get('clean_abstract', '')
            )
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to load all cleaned papers from the given directory
def GetAllData(years: List[int]) -> List[Paper]:
    """Returns a list of cleaned papers from the specified years."""
    all_papers = []
    
    for year in years:
        cleaned_dir = f'./data/processed/{year}'
        if os.path.exists(cleaned_dir):
            filenames = [f for f in os.listdir(cleaned_dir) if os.path.isfile(os.path.join(cleaned_dir, f))]
            print(f"Found {len(filenames)} files in {cleaned_dir}")
            
            for filename in filenames:
                file_path = os.path.join(cleaned_dir, filename)
                cleaned_paper = get_cleaned_paper(file_path)
                if cleaned_paper:
                    all_papers.append(cleaned_paper)
        else:
            print(f"Directory for year {year} does not exist.")
    
    return all_papers

# Example usage
#if __name__ == "__main__":
    #years = [2018, 2019, 2020, 2021, 2022, 2023]  # Define the range of years
    #papers = GetAllData(years)
    
    # Print out some example papers
    #for paper in papers[:5]:  # Show the first 5 papers as a sample
        #print(f"Title: {paper.title}\nAbstract: {paper.abstract}\nKeywords: {paper.keywords}\n")
