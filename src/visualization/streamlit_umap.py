import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from faker import Faker
import plotly.graph_objects as go
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer
import networkx as nx
from transformers import AutoTokenizer, AutoModel
import torch
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_model.predict.predict import *
from data_engineering.data import *

# Use mock or not
use_mock = False

@st.cache_data
def load_mock_data():
    faker = Faker()
    data = []
    codes = [f'Class{i}' for i in range(100)]
    for _ in range(500): 
        data.append({
            'Title': faker.sentence(nb_words=6),
            'abstract': faker.paragraph(nb_sentences=5),
            'subject_area': faker.word(),
            'Keywords': faker.words(nb=5)
        })
    df = pd.DataFrame(data)
    return df

@st.cache_data
def load_actual_data():
    # Load the actual data
    papers = GetCleanedData()
    df = pd.DataFrame(papers)
    return df

@st.cache_data
def load_predicted_data(df):
    # Load the predicted data
    config = {
        "model_save_path": "./models/LogisticRegression_single_hyper_01/2018_2023/single_label_classifier.pkl",
        "label_encoder_save_path": "./models/LogisticRegression_single_hyper_01/2018_2023/label_encoder.pkl",
        "tokenizer_model_save_dir": "./models/LogisticRegression_single_hyper_01/2018_2023/tokenizer_model/",
        "batch_size": 16,
        'tokenizer_model_name': 'allenai/scibert_scivocab_uncased'
    }
    predicted_df = predict_new_data(df, config)
    return predicted_df

def preprocess_data(df, p_flag):
    # Extract unique codes
    sarea = 'subject_area' if not p_flag else 'predicted_subject_area'
    unique_codes = sorted(df[sarea].unique())
    lb = LabelBinarizer()
    code_matrix = lb.fit_transform(df[sarea])
    return code_matrix, unique_codes

def generate_combined_embeddings(df, code_matrix):
    # Load pre-trained transformer model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # You can use any other model
    model = AutoModel.from_pretrained('bert-base-uncased')
    
    # Function to encode the abstracts using the transformer model
    def encode_abstracts(abstracts):
        inputs = tokenizer(abstracts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Return the embeddings (last hidden state of the model)
        return outputs.last_hidden_state.mean(dim=1).numpy()  # Averaging the token embeddings to get a single vector per abstract
    
    # Generate embeddings for the abstracts
    abstract_embeddings = encode_abstracts(df['abstract'].tolist())
    
    # Combine Abstract embeddings with the subject_area matrix
    combined_embeddings = np.hstack((abstract_embeddings, code_matrix))
    return combined_embeddings

def cluster_and_reduce(embeddings, n_clusters=10):
    # Dimensionality reduction and clustering
    umap_reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=-1)
    reduced_data = umap_reducer.fit_transform(embeddings)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # cluster_labels = kmeans.fit_predict(reduced_data)
    cluster_labels = []
    return reduced_data, cluster_labels

def compute_error_analysis(actual_df, predicted_df):
    results = []
    for i, (actual_codes, predicted_codes) in enumerate(zip(actual_df['subject_area'], predicted_df['predicted_subject_area'])):
        actual_set = set([actual_codes])
        predicted_set = set([predicted_codes])

        # Compute Jaccard similarity
        jaccard = len(actual_set & predicted_set) / len(actual_set | predicted_set) if len(actual_set | predicted_set) > 0 else 0

        # Store results
        results.append({
            'Paper ID': i,
            'Actual Subject Area': actual_codes,
            'Predicted Subject Area': predicted_codes,
            'Result': jaccard
        })
    return pd.DataFrame(results)

def plot_error_analysis(errors_df):
    fig = px.histogram(
        errors_df, 
        x='Result', 
        title='Distribution of Predicted Results',
        nbins=20,
        labels={'x': 'Result'},
        height=600
    )
    fig.update_layout(xaxis_title='Result', yaxis_title='Count')
    return fig

def plot_3d_scatter_with_labels(reduced_data, clusters, labels, hover_labels):
    # Map cluster numbers to human-readable labels
    cluster_map = {i: f"cluster_{i+1}" for i in np.unique(labels)}
    cluster_labels = [cluster_map[label] for label in labels]

    # Create a DataFrame for plotting
    umap_df = pd.DataFrame({
        'UMAP Dimension 1': reduced_data[:, 0],
        'UMAP Dimension 2': reduced_data[:, 1],
        'UMAP Dimension 3': reduced_data[:, 2],
        # 'Cluster': cluster_labels,
        'Classes': hover_labels
    })

    # Plot with Plotly
    fig = px.scatter_3d(
        umap_df,
        x='UMAP Dimension 1',
        y='UMAP Dimension 2',
        z='UMAP Dimension 3',
        # color='Cluster',  # Color by cluster
        hover_data=['Classes'],  # Hover details
        category_orders={'Cluster': list(cluster_map.values())}  # Order clusters
    )

    # Update layout for clarity
    fig.update_layout(
        legend_title="Clusters",
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        )
    )
    return fig

def plot_class_distribution(df, num):
    # Count the number of instances in each class
    class_counts = df['subject_area'].value_counts().reset_index().head(num)
    class_counts.columns = ['Class', 'Count']

    # Compute the percentage
    total_count = class_counts['Count'].sum()
    class_counts['Percentage'] = (class_counts['Count'] / total_count) * 100

    # Create a horizontal bar chart
    fig = px.bar(class_counts, y='Class', x='Percentage', title='Subject Area Distribution', height=600, color='Class', orientation='h')
    fig.update_layout(xaxis_title='Percentage', yaxis_title='Subject Areas')
    return fig

def main():
    st.set_page_config(page_title="ClassiPaper", layout="wide")

    st.title("Paper Research Classification Prediction Visualization")

    st.write("")
    st.write("")

    # Load data in session state
    if 'actual_df' not in st.session_state:
        st.session_state['actual_df'] = load_mock_data() if use_mock else load_actual_data()
        st.session_state['code_matrix'], st.session_state['unique_codes'] = preprocess_data(st.session_state['actual_df'], False)
        st.session_state['embeddings'] = generate_combined_embeddings(
            st.session_state['actual_df'], st.session_state['code_matrix']
        )
        st.session_state['reduced_data'], st.session_state['cluster_labels'] = cluster_and_reduce(st.session_state['embeddings'])

    if 'predicted_df' not in st.session_state:
        st.session_state['predicted_df'] = load_mock_data() if use_mock else load_predicted_data(st.session_state['actual_df'])
        st.session_state['p_code_matrix'], st.session_state['p_unique_codes'] = preprocess_data(st.session_state['predicted_df'], True)
        st.session_state['p_embeddings'] = generate_combined_embeddings(
            st.session_state['predicted_df'], st.session_state['p_code_matrix']
        )
        st.session_state['p_reduced_data'], st.session_state['p_cluster_labels'] = cluster_and_reduce(st.session_state['p_embeddings'])

    # Load data from session state
    df = st.session_state['actual_df']
    predicted_df = st.session_state['predicted_df']
    code_matrix = st.session_state['code_matrix']
    unique_codes = st.session_state['unique_codes']
    p_code_matrix = st.session_state['p_code_matrix']
    p_unique_codes = st.session_state['p_unique_codes']

    # Load clustering results from session state
    reduced_data = st.session_state['reduced_data']
    p_reduced_data = st.session_state['p_reduced_data']
    cluster_labels = st.session_state['cluster_labels']
    p_cluster_labels = st.session_state['p_cluster_labels']

    # Sidebar
    st.sidebar.title("Settings")

    # UMAP clustering visualization
    st.header("UMAP Clustering Visualization")

    # umap actual and predicted data cols
    col1, col2 = st.columns(2)

    with col1:
        # UMAP 3D Scatter Plot (actual)
        fig_scatter = plot_3d_scatter_with_labels(reduced_data, unique_codes, cluster_labels, df['subject_area'])
        fig_scatter.update_layout(height=650, title="3D UMAP Clustering of Actual Data")
        st.plotly_chart(fig_scatter)

    with col2:
        # UMAP 3D Scatter Plot (predicted)
        fig_scatter = plot_3d_scatter_with_labels(p_reduced_data, p_unique_codes, p_cluster_labels, predicted_df['predicted_subject_area'])
        fig_scatter.update_layout(height=650, title="3D UMAP Clustering of Predicted Data")
        st.plotly_chart(fig_scatter)

    show_data = st.checkbox("Datasets Preview")

    if show_data:
        dat1, dat2 = st.columns(2)
        with dat1:
            st.subheader("Actual Data")
            st.write(df)
        
        with dat2:
            st.subheader("Predicted Data")
            st.write(predicted_df)

    st.markdown("---")

    # Class distribution settings
    st.sidebar.subheader("Class Distribution Settings")
    num_classes = st.sidebar.slider("Number of Classes", min_value=10, max_value=len(unique_codes), value=10)
    st.sidebar.markdown("---")

    # Class distribution bar chart
    st.header("Subject Area Distribution")
    fig_class_dist = plot_class_distribution(df, num_classes)
    st.plotly_chart(fig_class_dist)

    st.markdown("---")

    # Error analysis settings
    st.sidebar.subheader("Error Analysis Settings")
    if st.sidebar.checkbox("Show Error Analysis", value=True):
        st.header("Error Analysis Between Actual and Predicted Data")

        st.write("")
        st.write("")
        st.write("")

        # Compute error analysis
        errors_df = compute_error_analysis(df, predicted_df)

        # Display a summary of error metrics
        st.subheader("Error Metrics Summary")
        avg_jaccard = errors_df['Result'].mean()
        st.subheader(f"**Average Jaccard Similarity:** {avg_jaccard:.2f}")

        st.write("")
        st.write("")

        # Plot error distribution
        st.subheader("Error Distribution")
        fig_error_dist = plot_error_analysis(errors_df)
        st.plotly_chart(fig_error_dist)

        # Show detailed error table
        st.subheader("Detailed Error Analysis")
        st.dataframe(errors_df)


if __name__ == "__main__":
    main()