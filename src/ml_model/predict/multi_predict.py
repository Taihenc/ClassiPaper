# predict_model.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import re
import nltk
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pickle

# ===========================
# Download necessary NLTK data
# ===========================
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def clean_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Cleans and preprocesses the input text.

    Args:
        text (str): The text to clean.
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer.
        stop_words (set): Set of stopwords.

    Returns:
        str: Cleaned and lemmatized text.
    """
    if not isinstance(text, str):
        return ''
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def preprocess_new_data(df: pd.DataFrame, lemmatizer: WordNetLemmatizer, stop_words: set) -> pd.DataFrame:
    """
    Preprocesses the new data by cleaning text and combining relevant fields.

    Args:
        df (pd.DataFrame): DataFrame containing new data with 'title', 'abstract', and 'keywords'.
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer.
        stop_words (set): Set of stopwords.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with a 'combined_text' column.
    """
    df['clean_title'] = df['title'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    df['clean_abstract'] = df['abstract'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    df['clean_keywords'] = df['keywords'].apply(
        lambda kws: clean_text(' '.join(kws), lemmatizer, stop_words) if isinstance(kws, list) else ''
    )
    df['combined_text'] = df['clean_title'].astype(str) + ' ' + df['clean_abstract'].astype(str) + ' ' + df['clean_keywords'].astype(str)
    df = df[df['combined_text'].str.strip() != '']  # Remove empty combined_text
    return df


def load_artifacts(config: dict):
    """
    Loads the saved model pipeline, MultiLabelBinarizer, tokenizer, embedding model, and thresholds.

    Args:
        config (dict): Configuration dictionary containing paths to saved artifacts.

    Returns:
        tuple: (model_pipeline, mlb, tokenizer, embedding_model, device, thresholds)
    """
    # Load Model Pipeline
    if not os.path.exists(config['model_save_path']):
        print(f"Model pipeline file not found at: {config['model_save_path']}")
        raise FileNotFoundError(f"Model pipeline file not found at: {config['model_save_path']}")
    model_pipeline = joblib.load(config['model_save_path'])
    print(f"Model pipeline loaded from {config['model_save_path']}")

    # Load MultiLabelBinarizer
    if not os.path.exists(config['mlb_save_path']):
        print(f"MultiLabelBinarizer file not found at: {config['mlb_save_path']}")
        raise FileNotFoundError(f"MultiLabelBinarizer file not found at: {config['mlb_save_path']}")
    mlb = joblib.load(config['mlb_save_path'])
    print(f"MultiLabelBinarizer loaded from {config['mlb_save_path']}")

    # Load Tokenizer and Embedding Model
    if not os.path.exists(config['tokenizer_model_save_dir']):
        print(f"Tokenizer/model directory not found at: {config['tokenizer_model_save_dir']}. Downloading instead.")
        model_name_or_path = config['tokenizer_model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        embedding_model = AutoModel.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_model_save_dir'])
        embedding_model = AutoModel.from_pretrained(config['tokenizer_model_save_dir'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model.to(device)
    embedding_model.eval()
    print(f"Tokenizer and embedding model loaded or downloaded successfully. Device: {device}")


    # Load Thresholds
    if not os.path.exists(config['thresholds_save_path']):
        print(f"Thresholds file not found at: {config['thresholds_save_path']}")
        raise FileNotFoundError(f"Thresholds file not found at: {config['thresholds_save_path']}")
    with open(config['thresholds_save_path'], 'rb') as f:
        thresholds = pickle.load(f)
    print(f"Thresholds loaded from {config['thresholds_save_path']}")

    return model_pipeline, mlb, tokenizer, embedding_model, device, thresholds


def generate_embeddings(text_list: List[str], tokenizer: AutoTokenizer, model: AutoModel, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """
    Generates CLS token embeddings for a list of texts.

    Args:
        text_list (List[str]): List of text strings.
        tokenizer (AutoTokenizer): Tokenizer for encoding text.
        model (AutoModel): Pre-trained embedding model.
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size for processing.

    Returns:
        np.ndarray: Array of embeddings.
    """
    embeddings = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    preprocessed_texts = [clean_text(text, lemmatizer, stop_words) for text in text_list]
    valid_indices = [i for i, text in enumerate(preprocessed_texts) if text.strip()]
    valid_texts = [preprocessed_texts[i] for i in valid_indices]

    if not valid_texts:
        print("No valid texts to generate embeddings.")
        return np.array([])

    with torch.no_grad():
        for i in tqdm(range(0, len(valid_texts), batch_size), desc="Generating Embeddings"):
            batch_text = valid_texts[i:i+batch_size]
            encoded_input = tokenizer.batch_encode_plus(
                batch_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            outputs = model(**encoded_input)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(embeddings)


def predict_new_data(new_data: pd.DataFrame, config: dict):
    """
    Processes new data and makes predictions using the saved model pipeline.

    Args:
        new_data (pd.DataFrame): DataFrame containing new data with 'title', 'abstract', and 'keywords'.
        config (dict): Configuration dictionary containing paths to saved artifacts.

    Returns:
        pd.DataFrame: DataFrame with predictions and confidence scores.
    """
    # Initialize preprocessing tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Preprocess new data
    preprocessed_df = preprocess_new_data(new_data, lemmatizer, stop_words)

    # Load artifacts
    model_pipeline, mlb, tokenizer, embedding_model, device, thresholds = load_artifacts(config)

    # Generate embeddings
    embeddings = generate_embeddings(preprocessed_df['combined_text'].tolist(), tokenizer, embedding_model, device, batch_size=config.get('batch_size', 32))

    if embeddings.size == 0:
        print("No embeddings generated. Check input data.")
        return pd.DataFrame()

    # Make prediction scores
    y_scores = model_pipeline.predict_proba(embeddings)

    # Apply thresholds to get binary predictions
    y_pred = (y_scores >= thresholds).astype(int)

    # Decode predictions to get label names
    decoded_predictions = mlb.inverse_transform(y_pred)

    # Extract confidence scores for predicted labels
    confidence_scores = []
    for idx, preds in enumerate(y_pred):
        scores = y_scores[idx]
        labels = mlb.classes_
        pred_confidences = {}
        for i, label in enumerate(labels):
            if preds[i] == 1:
                pred_confidences[label] = scores[i]
        confidence_scores.append(pred_confidences)

    # Prepare the result DataFrame
    result_df = preprocessed_df.copy()
    result_df['predicted_subject_area'] = decoded_predictions
    result_df['confidence_scores'] = confidence_scores

    return result_df[['title', 'abstract', 'keywords', 'predicted_subject_area', 'confidence_scores']]


if __name__ == "__main__":
    # Define the configuration (ensure paths are correct)
    config = {
        "model_save_path": "./models/LogisticRegression_04/2018_2023/multi_label_classifier.pkl",
        "mlb_save_path": "./models/LogisticRegression_04/2018_2023/mlb.pkl",
        "tokenizer_model_save_dir": "./models/LogisticRegression_04/2018_2023/tokenizer_model/",
        "thresholds_save_path": "./models/LogisticRegression_04/2018_2023/thresholds.pkl",
        "batch_size": 16,
        'tokenizer_model_name': "allenai/scibert_scivocab_uncased"
    }

    # Example new data (modify as needed)
    data = {
        'title': [
            "Utility-based cache partitioning: A low-overhead, high-performance, runtime mechanism to partition shared caches",
            "Advancements in Quantum Computing"
        ],
        'abstract': [
            "This paper investigates the problem of partitioning a shared cache between multiple concurrently executing applications. The commonly used LRU policy implicitly partitions a shared cache on a demand basis, giving more cache resources to the application that has a high demand and fewer cache resources to the application that has a low demand. However, a higher demand for cache resources does not always correlate with a higher performance from additional cache resources. It is beneficial for performance to invest cache resources in the application that benefits more from the cache resources rather than in the application that has more demand for the cache resources. This paper proposes utility-based cache partitioning (UCP), a low-overhead, runtime mechanism that partitions a shared cache between multiple applications depending on the reduction in cache misses that each application is likely to obtain for a given amount of cache resources. The proposed mechanism monitors each application at runtime using a novel, cost-effective, hardware circuit that requires less than 2kB of storage. The information collected by the monitoring circuits is used by a partitioning algorithm to decide the amount of cache resources allocated to each application. Our evaluation, with 20 multiprogrammed workloads, shows that UCP improves performance of a dual-core system by up to 23% and on average 11% over LRU-based cache partitioning",
            "Quantum computing has emerged as a revolutionary technology with the potential to solve complex problems that are intractable for classical computers. Recent advancements have focused on improving qubit stability, error correction, and scalable architectures. This paper reviews the current state of quantum computing and explores future directions for research and development."
        ],
        'keywords': [
            ["Runtime", "Application software"],
            ["Quantum Computing", "Qubits", "Error Correction"]
        ]
    }

    new_data_df = pd.DataFrame(data)

    # Make predictions
    predictions_df = predict_new_data(new_data_df, config)

    # Display predictions
    print("\nPredictions:")
    print(predictions_df)

    # print each prediction with confidence scores
    for idx, row in predictions_df.iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Abstract: {row['abstract']}")
        print(f"Keywords: {row['keywords']}")
        print(f"Predicted Subject Areas: {row['predicted_subject_area']}")
        print("Confidence Scores:")
        for label, score in row['confidence_scores'].items():
            print(f"  - {label}: {score}")
