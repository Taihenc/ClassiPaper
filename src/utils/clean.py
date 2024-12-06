import os
import json
import pandas as pd
import re
import logging
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the base directory paths
base_data_dir = r'.\data\raw-provided\Data 2018-2023\Project'
base_cleaned_dir = r'.\data\processed'

# List of years to process
years = [str(year) for year in range(2018, 2024)]

# Data Cleaning Functions
def clean_text(text):
    logging.debug("Cleaning text.")
    if not isinstance(text, str):
        return ''  # Return empty string if not a valid text
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove stop words using sklearn's built-in stop words
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    # Rejoin and return cleaned text
    return ' '.join(tokens)

def clean_keywords(keywords):
    logging.debug("Cleaning keywords.")
    if not isinstance(keywords, list):
        return ''
    # Remove duplicates and stopwords from keywords
    cleaned_keywords = list(set([kw for kw in keywords if kw not in ENGLISH_STOP_WORDS and len(kw) > 1]))
    return ' '.join(cleaned_keywords)

def clean_title(title):
    logging.debug("Cleaning title.")
    if not isinstance(title, str) or not title.strip():
        return ''  # Return empty string if title is empty or not a valid string
    # Fix title case (capitalize first letter of each word)
    title = title.title()
    # Remove common leading articles or words
    common_words = ['a', 'an', 'the']
    title_words = title.split()
    if title_words and title_words[0].lower() in common_words:
        title = ' '.join(title_words[1:])
    return title

def clean_abstract(abstract):
    logging.debug("Cleaning abstract.")
    if not isinstance(abstract, str):
        return ''
    # Remove strange characters and fix encoding issues
    abstract = re.sub(r'[^\x00-\x7F]+', '', abstract)  # Remove non-ASCII characters
    abstract = abstract.strip()  # Remove leading/trailing spaces
    return abstract

# Loop through each year folder and process files
for year in years:
    data_dir = os.path.join(base_data_dir, year)
    cleaned_dir = os.path.join(base_cleaned_dir, year)
    
    # Create the cleaned directory if it doesn't exist
    os.makedirs(cleaned_dir, exist_ok=True)

    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: The directory {data_dir} does not exist.")
    else:
        filenames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        print(f"Found {len(filenames)} files to load for {year}.")

        data = []
        
        def load_file(filename):
            print(f"Loading file: {filename}")
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return json.load(file)  # Parse JSON
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                return None  # Handle errors gracefully

        # Load files in parallel for efficiency
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_file, filenames))
            data = [item for item in results if item is not None]

        # Initialize lists to hold data
        titles = []
        abstracts = []
        keywords = []
        subject_areas = []

        # Extract relevant fields from the data
        for item in data:
            try:
                # Title
                titles.append(item.get('abstracts-retrieval-response', {}).get('coredata', {}).get('dc:title', ''))
                
                # Abstract
                abstracts.append(item.get('abstracts-retrieval-response', {}).get('coredata', {}).get('dc:description', ''))
                
                # Keywords
                authkeywords = item.get('abstracts-retrieval-response', {}).get('authkeywords', {})
                if isinstance(authkeywords, dict):
                    author_keywords = authkeywords.get('author-keyword', [])
                    if isinstance(author_keywords, list):
                        keywords.append([kw.get('$', '') for kw in author_keywords])
                    elif isinstance(author_keywords, dict):
                        keywords.append([author_keywords.get('$', '')])
                    else:
                        keywords.append([])  # Empty if no valid keywords
                else:
                    keywords.append([])  # Empty if no valid keywords
                
                # Subject Areas
                subject_area_list = item.get('abstracts-retrieval-response', {}).get('subject-areas', {}).get('subject-area', [])
                subject_areas.append([area.get('$', '') for area in subject_area_list])

            except Exception as e:
                logging.error(f"Error processing item: {e}")

        # Create a DataFrame
        df = pd.DataFrame({
            'title': titles,
            'abstract': abstracts,
            'keywords': keywords,
            'subject_areas': subject_areas
        })

        # Apply the cleaning functions efficiently
        df['clean_title'] = df['title'].apply(clean_title)
        df['clean_abstract'] = df['abstract'].apply(clean_abstract)
        df['clean_keywords'] = df['keywords'].apply(lambda kws: clean_keywords(kws) if isinstance(kws, list) else '')

        # Save cleaned data to the new directory
        for i, filename in enumerate(filenames):
            cleaned_data = {
                'clean_title': df.iloc[i]['clean_title'],
                'clean_abstract': df.iloc[i]['clean_abstract'],
                'clean_keywords': df.iloc[i]['clean_keywords'],
            }
            
            # Save to JSON file
            cleaned_filename = os.path.join(cleaned_dir, f'cleaned_{filename}')
            try:
                with open(cleaned_filename, 'w', encoding='utf-8') as file:
                    json.dump(cleaned_data, file, ensure_ascii=False, indent=4)
                print(f"Saved cleaned data for {filename} to {cleaned_filename}")
            except Exception as e:
                print(f"Failed to save {filename}: {e}")

        print(f"Cleaning and saving completed for {year}.")
