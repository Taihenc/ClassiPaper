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

# Data Cleaning Functions
def clean_text(text):
    if not isinstance(text, str):
        return ''  # Return empty string if not a valid text
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Lowercase
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

def clean_keywords(keywords):
    if not isinstance(keywords, list):
        return ''
    cleaned_keywords = list(set([kw for kw in keywords if kw not in ENGLISH_STOP_WORDS and len(kw) > 1]))
    return ' '.join(cleaned_keywords)

def clean_title(title):
    if not isinstance(title, str) or not title.strip():
        return ''
    title = title.title()
    common_words = ['a', 'an', 'the']
    title_words = title.split()
    if title_words and title_words[0].lower() in common_words:
        title = ' '.join(title_words[1:])
    return title

def clean_abstract(abstract):
    if not isinstance(abstract, str):
        return ''
    abstract = re.sub(r'[^\x00-\x7F]+', '', abstract)
    return abstract.strip()

# Combine data across all directories
combined_data = []

# Process all files in the directories specified in BASE_DATA_DIR
for data_dir in base_data_dirs:
    if not os.path.exists(data_dir):
        logging.warning(f"Warning: The directory {data_dir} does not exist.")
        continue

    filenames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    logging.info(f"Processing {len(filenames)} files in {data_dir}.")

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
        results = list(executor.map(load_file, filenames))
        data = [item for item in results if item is not None]

    # Process and clean data
    dir_cleaned_count = 0
    for item in data:
        try:
            title = clean_title(item.get('abstracts-retrieval-response', {}).get('coredata', {}).get('dc:title', ''))
            abstract = clean_abstract(item.get('abstracts-retrieval-response', {}).get('coredata', {}).get('dc:description', ''))

            keywords = []
            authkeywords = item.get('abstracts-retrieval-response', {}).get('authkeywords', {})
            if isinstance(authkeywords, dict):
                author_keywords = authkeywords.get('author-keyword', [])
                if isinstance(author_keywords, list):
                    keywords = [kw.get('$', '') for kw in author_keywords]
                elif isinstance(author_keywords, dict):
                    keywords = [author_keywords.get('$', '')]
            cleaned_keywords = clean_keywords(keywords)

            combined_data.append({
                'clean_title': title,
                'clean_abstract': abstract,
                'clean_keywords': cleaned_keywords,
            })
            dir_cleaned_count += 1
        except Exception as e:
            logging.error(f"Error processing item: {e}")

    logging.info(f"Completed processing {dir_cleaned_count} files from {data_dir}.")

# Save all data into a single combined file
os.makedirs(base_cleaned_dir, exist_ok=True)
combined_filename = os.path.join(base_cleaned_dir, 'cleaned_data_combined.json')

try:
    with open(combined_filename, 'w', encoding='utf-8') as file:
        json.dump(combined_data, file, ensure_ascii=False, indent=4)
    logging.info(f"Saved combined cleaned data to {combined_filename}")
except Exception as e:
    logging.error(f"Failed to save combined cleaned data: {e}")
