import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import plotly.express as px
from faker import Faker
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MultiLabelBinarizer
import plotly.figure_factory as ff
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import jaccard_score
from joblib import Parallel, delayed
from joblib import Memory

memory = Memory("cache_directory", verbose=0)

# Use mock or not
use_mock = True

def load_mock_data():
    faker = Faker()
    data = []
    codes = [f'Class{i}' for i in range(100)]
    for _ in range(500): 
        data.append({
            'Title': faker.sentence(nb_words=6),
            'Abstract': faker.paragraph(nb_sentences=5),
            'Classification Codes': faker.random_choices(elements=codes, length=4),
            'Date of Publication': faker.date_this_decade(),
            'Affiliations': faker.company(),
            'Citations': faker.random_int(min=0, max=100),
            'Keywords': faker.words(nb=5)
        })
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    # Extract unique codes
    unique_codes = sorted(set(code for codes in df['Classification Codes'] for code in codes))
    mlb = MultiLabelBinarizer(classes=unique_codes)
    code_matrix = mlb.fit_transform(df['Classification Codes'])
    return code_matrix, unique_codes

@memory.cache
def generate_embeddings(df):
    # Mock embedding generation (replace with actual model like Sentence-BERT)
    vectorizer = TfidfVectorizer(max_features=100)
    abstract_embeddings = vectorizer.fit_transform(df['Abstract']).toarray()
    return abstract_embeddings

def cluster_and_reduce(embeddings, n_clusters=10):
    # Dimensionality reduction and clustering
    umap_reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=-1)
    reduced_data = umap_reducer.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(reduced_data)
    return reduced_data, cluster_labels

@memory.cache
def perform_clustering(embeddings, n_clusters=10):
    return cluster_and_reduce(embeddings, n_clusters)

# def precompute_clusters(embeddings, max_clusters=20):
#     cluster_results = {}
#     for n_clusters in range(2, max_clusters + 1):
#         reduced_data, cluster_labels = cluster_and_reduce(embeddings, n_clusters)
#         cluster_results[n_clusters] = (reduced_data, cluster_labels)
#     return cluster_results

def compute_jaccard_similarity(actual_codes, predicted_codes):
    actual_set = set(actual_codes)
    predicted_set = set(predicted_codes)
    return len(actual_set & predicted_set) / len(actual_set | predicted_set) if len(actual_set | predicted_set) > 0 else 0

def compute_error_analysis_parallel(actual_df, predicted_df):
    results = Parallel(n_jobs=-1)(delayed(lambda i, row: {
        'Paper ID': i,
        'Actual Codes': row[0],
        'Predicted Codes': row[1],
        'Jaccard Similarity': compute_jaccard_similarity(row[0], row[1])
    })(i, (row[0], row[1])) for i, row in enumerate(zip(actual_df['Classification Codes'], predicted_df['Classification Codes'])))
    
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

    # Create a bar chart
    fig = px.bar(class_counts, x='Class', y='Count', title='Class Distribution', height=600, color='Class')
    fig.update_layout(xaxis_title='Classification Code', yaxis_title='Number of Instances')
    return fig

def main():
    st.set_page_config(page_title="ClassiPaper", layout="wide")

    st.title("Paper Research Classification Prediction Clustering and Visualization")

    st.write("")
    st.write("")

    # Sidebar
    st.sidebar.title("Settings")
    # st.sidebar.subheader("Clustering Settings")
    # n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=20, value=10)
    st.sidebar.subheader("Class Distribution Settings")
    num_classes = st.sidebar.slider("Number of Classes", min_value=10, max_value=100, value=25)

    # Load data in session state
    if 'actual_df' not in st.session_state:
        st.session_state['actual_df'] = load_mock_data() if use_mock else None
        st.session_state['code_matrix'], st.session_state['unique_codes'] = preprocess_data(st.session_state['actual_df'])
        st.session_state['embeddings'] = generate_embeddings(st.session_state['actual_df'])
        st.session_state['reduced_data'], st.session_state['cluster_labels'] = perform_clustering(st.session_state['embeddings'])
        # st.session_state['cluster_results'] = precompute_clusters(st.session_state['embeddings'])

    if 'predicted_df' not in st.session_state:
        st.session_state['predicted_df'] = load_mock_data() if use_mock else None
        st.session_state['p_code_matrix'], st.session_state['p_unique_codes'] = preprocess_data(st.session_state['predicted_df'])
        st.session_state['p_embeddings'] = generate_embeddings(st.session_state['predicted_df'])
        st.session_state['p_reduced_data'], st.session_state['p_cluster_labels'] = perform_clustering(st.session_state['p_embeddings'])
        # st.session_state['p_cluster_results'] = precompute_clusters(st.session_state['p_embeddings'])

    # Load data from session state
    df = st.session_state['actual_df']
    predicted_df = st.session_state['predicted_df']
    code_matrix = st.session_state['code_matrix']
    unique_codes = st.session_state['unique_codes']
    p_code_matrix = st.session_state['p_code_matrix']
    p_unique_codes = st.session_state['p_unique_codes']
    
    # Generate embeddings
    # embeddings = st.session_state['embeddings']
    # p_embeddings = st.session_state['p_embeddings']

    # Perform clustering and dimensionality reduction
    reduced_data, cluster_labels = st.session_state['reduced_data'], st.session_state['cluster_labels']
    p_reduced_data, p_cluster_labels = st.session_state['p_reduced_data'], st.session_state['p_cluster_labels']
    # reduced_data, cluster_labels = st.session_state['cluster_results'][n_clusters]
    # p_reduced_data, p_cluster_labels = st.session_state['p_cluster_results'][n_clusters]

    # UMAP clustering visualization
    st.header("UMAP Clustering Visualization")

    # umap actual and predicted data cols
    col1, col2 = st.columns(2)

    with col1:
        # UMAP 3D Scatter Plot (actual)
        fig_scatter = plot_3d_scatter_with_labels(reduced_data, unique_codes, cluster_labels, df['Classification Codes'])
        fig_scatter.update_layout(height=650, title="3D UMAP Clustering of Actual Data")
        st.plotly_chart(fig_scatter)

        show_actual_data = st.checkbox("Actual Dataset Preview")
        if show_actual_data:
            st.dataframe(df)

    with col2:
        # UMAP 3D Scatter Plot (predicted)
        fig_scatter = plot_3d_scatter_with_labels(p_reduced_data, p_unique_codes, p_cluster_labels, predicted_df['Classification Codes'])
        fig_scatter.update_layout(height=650, title="3D UMAP Clustering of Predicted Data")
        st.plotly_chart(fig_scatter)

        show_predicted_data = st.checkbox("Predicted Dataset Preview")
        if show_predicted_data:
            st.dataframe(predicted_df)

    st.markdown("---")

    # Class distribution bar chart
    st.header("Classification Code Distribution")
    fig_class_dist = plot_class_distribution(df, num_classes)
    st.plotly_chart(fig_class_dist)

    st.markdown("---")

    if st.sidebar.checkbox("Show Error Analysis", value=True):
        st.header("Error Analysis Between Actual and Predicted Data")

        st.write("")
        st.write("")

        # Compute error analysis
        errors_df = compute_error_analysis_parallel(df, predicted_df)

        # Display a summary of error metrics
        st.subheader("Error Metrics Summary")
        avg_jaccard = errors_df['Jaccard Similarity'].mean()
        st.write(f"**Average Jaccard Similarity:** {avg_jaccard:.2f}")

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