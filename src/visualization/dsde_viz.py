import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from faker import Faker
import plotly.graph_objects as go
from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import networkx as nx
from transformers import AutoTokenizer, AutoModel
import torch
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_model.predict.predict import *
from data_engineering.data import *

# Use mock or not
use_mock = True

@st.cache_data
def load_mock_data():
    faker = Faker()
    data = []
    codes = [f'Class{i}' for i in range(100)]
    for _ in range(500): 
        data.append({
            'Title': faker.sentence(nb_words=6),
            'Abstract': faker.paragraph(nb_sentences=5),
            'Classification Codes': faker.random_choices(elements=codes, length=4),
            'Keywords': faker.words(nb=5)
        })
    df = pd.DataFrame(data)
    return df

@st.cache_data
def load_actual_data():
    # Load the actual data
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    papers = GetAllData(years)
    df = pd.DataFrame([vars(paper) for paper in papers])
    return df

def preprocess_data(df):
    # Extract unique codes
    unique_codes = sorted(set(code for codes in df['Classification Codes'] for code in codes))
    mlb = MultiLabelBinarizer(classes=unique_codes)
    code_matrix = mlb.fit_transform(df['Classification Codes'])
    return code_matrix, unique_codes

# def generate_combined_embeddings(df, code_matrix):
#     # Generate embeddings from the Abstract field
#     vectorizer = TfidfVectorizer(max_features=100)
#     abstract_embeddings = vectorizer.fit_transform(df['Abstract']).toarray()
    
#     # Combine Abstract embeddings with the Classification Codes matrix
#     combined_embeddings = np.hstack((abstract_embeddings, code_matrix))
#     return combined_embeddings

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
    abstract_embeddings = encode_abstracts(df['Abstract'].tolist())
    
    # Combine Abstract embeddings with the Classification Codes matrix
    combined_embeddings = np.hstack((abstract_embeddings, code_matrix))
    return combined_embeddings

def cluster_and_reduce(embeddings, n_clusters=10):
    # Dimensionality reduction and clustering
    umap_reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=-1)
    reduced_data = umap_reducer.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_data)
    return reduced_data, cluster_labels

def compute_error_analysis(actual_df, predicted_df):
    results = []
    for i, (actual_codes, predicted_codes) in enumerate(zip(actual_df['Classification Codes'], predicted_df['Classification Codes'])):
        actual_set = set(actual_codes)
        predicted_set = set(predicted_codes)

        # Compute Jaccard similarity
        jaccard = len(actual_set & predicted_set) / len(actual_set | predicted_set) if len(actual_set | predicted_set) > 0 else 0

        # Store results
        results.append({
            'Paper ID': i,
            'Actual Codes': actual_codes,
            'Predicted Codes': predicted_codes,
            'Jaccard Similarity': jaccard
        })
    return pd.DataFrame(results)

def plot_error_analysis(errors_df):
    fig = px.histogram(
        errors_df, 
        x='Jaccard Similarity', 
        title='Distribution of Jaccard Similarities',
        nbins=20,
        labels={'x': 'Jaccard Similarity'},
        height=600
    )
    fig.update_layout(xaxis_title='Jaccard Similarity', yaxis_title='Count')
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
        'Cluster': cluster_labels,
        'Classes': hover_labels
    })

    # Plot with Plotly
    fig = px.scatter_3d(
        umap_df,
        x='UMAP Dimension 1',
        y='UMAP Dimension 2',
        z='UMAP Dimension 3',
        color='Cluster',  # Color by cluster
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
    class_counts = df['Classification Codes'].explode().value_counts().reset_index().head(num)
    class_counts.columns = ['Class', 'Count']

    # Compute the percentage
    total_count = class_counts['Count'].sum()
    class_counts['Percentage'] = (class_counts['Count'] / total_count) * 100

    # Create a horizontal bar chart
    fig = px.bar(class_counts, y='Class', x='Percentage', title='Class Distribution', height=600, color='Class', orientation='h')
    fig.update_layout(xaxis_title='Percentage', yaxis_title='Classification Code')
    return fig

def generate_cooccurrence_network(df):
    """Generate a co-occurrence graph of classification codes."""
    # Initialize the graph
    G = nx.Graph()

    # Add edges based on co-occurrence
    for codes in df['Classification Codes']:
        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                if G.has_edge(code1, code2):
                    G[code1][code2]['weight'] += 1
                else:
                    G.add_edge(code1, code2, weight=1)
    
    return G

def plot_network(G, top_n=None, node_size=None, show_edges=True, show_labels=True):
    """Plot a co-occurrence network using Plotly."""
    # Calculate degree centrality for sizing
    centrality = nx.degree_centrality(G)

    # Extract the top N nodes by degree centrality if specified
    if top_n:
        top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:top_n]
        G = G.subgraph(top_nodes)

    # Generate positions for nodes
    pos = nx.spring_layout(G, seed=42)

    # Create edge traces
    edge_x = []
    edge_y = []
    if show_edges:
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    hover_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # node_text.append(f"{node} (Degree: {G.degree[node]})")
        node_text.append(node)
        node_sizes.append(node_size * centrality[node] * 5)
        hover_text.append(f"{node} (Degree: {G.degree[node]})")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            size=node_sizes,
            color=node_sizes,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Centrality",
                xanchor='left',
                titleside='right'
            )
        ),
        text=node_text,
        textposition='top center' if show_labels else 'top center',
        hoverinfo='text',
        hovertext=hover_text
    )

    # Combine edge and node traces
    fig = go.Figure(data=[edge_trace, node_trace] if show_edges else [node_trace])
    fig.update_layout(
        title="Classification Codes Co-occurrence Network",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    return fig

def main():
    st.set_page_config(page_title="ClassiPaper", layout="wide")

    st.title("Paper Research Classification Prediction Clustering and Visualization")

    st.write("")
    st.write("")

    # Load data in session state
    if 'actual_df' not in st.session_state:
        st.session_state['actual_df'] = load_mock_data() if use_mock else None
        st.session_state['code_matrix'], st.session_state['unique_codes'] = preprocess_data(st.session_state['actual_df'])
        st.session_state['embeddings'] = generate_combined_embeddings(
            st.session_state['actual_df'], st.session_state['code_matrix']
        )
        st.session_state['reduced_data'], st.session_state['cluster_labels'] = cluster_and_reduce(st.session_state['embeddings'])

    if 'predicted_df' not in st.session_state:
        st.session_state['predicted_df'] = load_mock_data() if use_mock else None
        st.session_state['p_code_matrix'], st.session_state['p_unique_codes'] = preprocess_data(st.session_state['predicted_df'])
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
    reduced_data, cluster_labels = st.session_state['reduced_data'], st.session_state['cluster_labels']
    p_reduced_data, p_cluster_labels = st.session_state['p_reduced_data'], st.session_state['p_cluster_labels']

    # Sidebar
    st.sidebar.title("Settings")

    # UMAP clustering visualization
    st.header("UMAP Clustering Visualization")

    # umap actual and predicted data cols
    col1, col2 = st.columns(2)

    with col1:
        # UMAP 3D Scatter Plot (actual)
        fig_scatter = plot_3d_scatter_with_labels(reduced_data, unique_codes, cluster_labels, df['Classification Codes'])
        fig_scatter.update_layout(height=650, title="3D UMAP Clustering of Actual Data")
        st.plotly_chart(fig_scatter)

    with col2:
        # UMAP 3D Scatter Plot (predicted)
        fig_scatter = plot_3d_scatter_with_labels(p_reduced_data, p_unique_codes, p_cluster_labels, predicted_df['Classification Codes'])
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

    st.header("Classification Codes Network Analysis")

    # Generate and plot the co-occurrence network
    G = generate_cooccurrence_network(df)

    # Display network with top N nodes (optional)
    st.sidebar.subheader("Network Settings")
    top_n = st.sidebar.slider("Number of Top Nodes to Display", 10, len(G.nodes), value=20)
    node_size = st.sidebar.slider("Node Size", 1, 20, value=10)
    show_edges = st.sidebar.checkbox("Show Edges", value=True)
    show_labels = st.sidebar.checkbox("Show Class Names", value=True)
    st.sidebar.markdown("---")

    fig_network = plot_network(G, top_n=top_n, node_size=node_size, show_edges=show_edges, show_labels=show_labels)
    st.plotly_chart(fig_network)

    # Network statistics
    st.subheader("Network Statistics")
    st.write(f"**Number of Nodes:** {G.number_of_nodes()}")
    st.write(f"**Number of Edges:** {G.number_of_edges()}")
    st.write(f"**Average Degree:** {np.mean([d for n, d in G.degree()]):.2f}")

    st.markdown("---")

    # Class distribution settings
    st.sidebar.subheader("Class Distribution Settings")
    num_classes = st.sidebar.slider("Number of Classes", min_value=10, max_value=len(unique_codes), value=10)
    st.sidebar.markdown("---")

    # Class distribution bar chart
    st.header("Classification Code Distribution")
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
        avg_jaccard = errors_df['Jaccard Similarity'].mean()
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