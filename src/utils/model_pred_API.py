from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import os, sys

"""When Run you need to be in src/ml_model first ! ! !"""

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_model.predict.predict import *

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# CORS(app, resources={r"/pred": {"origins": "http://localhost:5173"}})

# Define the configuration
config = {
    "model_save_path": "./models/LogisticRegression_single_hyper_01/2018_2023/single_label_classifier.pkl",
    "label_encoder_save_path": "./models/LogisticRegression_single_hyper_01/2018_2023/label_encoder.pkl",
    "tokenizer_model_save_dir": "./models/LogisticRegression_single_hyper_01/2018_2023/tokenizer_model/",
    "batch_size": 16,
    'tokenizer_model_name': 'allenai/scibert_scivocab_uncased'
}

@app.route('/pred', methods=['POST'])
def pred():
    data = request.get_json()
    
    # Convert JSON data to DataFrame
    new_data_df = pd.DataFrame([data])
    
    # Make predictions
    predictions_df = predict_new_data(new_data_df, config)
    
    # Convert predictions to JSON
    result = predictions_df.to_dict(orient='records')[0]
    
    return jsonify(result)

if __name__ == '__main__':
    app.run()

"""When Run you need to be in src/ml_model first ! ! !"""
