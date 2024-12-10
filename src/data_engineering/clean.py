import os
import json
import re
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Set up logging
logging.basicConfig(level=logging.INFO)  # Change to INFO to reduce verbosity

# Load .env file
load_dotenv()

# Read environment variables
base_data_dirs = [path.strip().strip('"') for path in os.getenv("BASE_DATA_DIR", "").split(",")]
base_cleaned_dir = os.getenv("BASE_CLEANED_DIR", "").strip().strip('"')
base_cleaned_file = os.getenv("BASE_CLEANED_FILE", "").strip().strip('"')

# Data Cleaning Functions
def clean_text(text):
    if not isinstance(text, str):
        return ''  # Return empty string if not a valid text
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Lowercase
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# Combine data across all directories
combined_data = []

# Process all files in the directories specified in BASE_DATA_DIR
for data_dir in base_data_dirs:
    if not os.path.exists(data_dir):
        logging.warning(f"Warning: The directory {data_dir} does not exist.")
        continue

    filename = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    data = []

    def load_file(filename):
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)  # Parse JSON
        except Exception as e:
            logging.error(f"Failed to load {filename}: {e}")
            return None

    # Load files in parallel for efficiency
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_file, filename))
        data = [item for item in results if item is not None]

    # Process and clean data
    dir_cleaned_count = 0
    for item in data:
        try:
            # Extract and clean title and abstract
            title = clean_text(item.get('abstracts-retrieval-response', {}).get('coredata', {}).get('dc:title', ''))
            abstract = clean_text(item.get('abstracts-retrieval-response', {}).get('coredata', {}).get('dc:description', ''))

            # Process keywords
            keywords = []
            authkeywords = item.get('abstracts-retrieval-response', {}).get('authkeywords', {})
            if isinstance(authkeywords, dict):
                author_keywords = authkeywords.get('author-keyword', [])
                if isinstance(author_keywords, list):
                    keywords = [kw.get('$', '') for kw in author_keywords]
                elif isinstance(author_keywords, dict):
                    keywords = [author_keywords.get('$', '')]
            cleaned_keywords = ' '.join(clean_text(kw) for kw in keywords)

            # Process subject areas
            subject_areas = []
            subareas = item.get('abstracts-retrieval-response', {}).get('subject-areas', {})
            if isinstance(subareas, dict):
                subject_areas = subareas.get('subject-area', [])
                if isinstance(subject_areas, list):
                    subject_areas = [sa.get('$', '') for sa in subject_areas]
                elif isinstance(subject_areas, dict):
                    subject_areas = [subject_areas.get('$', '')]
            # pick first subject_area that end with '(all)
            filtered_subject_areas = [sa for sa in subject_areas if sa.endswith('(all)')]
            if len(filtered_subject_areas) == 0:
                continue
            cleaned_subject_areas = filtered_subject_areas[0]

            # Append processed data
            combined_data.append({
                'clean_title': title,
                'clean_abstract': abstract,
                'clean_keywords': cleaned_keywords,
                'clean_subject_area': cleaned_subject_areas
            })
            dir_cleaned_count += 1
        except Exception as e:
            logging.error(f"Error processing item: {e}")

    logging.info(f"Completed processing {dir_cleaned_count} files from {data_dir}.")

# Save all data into a single combined file
os.makedirs(base_cleaned_dir, exist_ok=True)

try:
    filename = os.path.join(base_cleaned_dir, base_cleaned_file)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved combined cleaned data to {filename}")
except Exception as e:
    logging.error(f"Failed to save combined cleaned data: {e}")
