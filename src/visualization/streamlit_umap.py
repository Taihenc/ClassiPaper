import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import plotly.express as px
from faker import Faker
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Use mock or not
use_mock = True

@st.cache_data
def load_mock_data():
    faker = Faker()
    # Generate mock data
    data = []
    codes = [f'Class{i}' for i in range(100)]
    for _ in range(50): 
        data.append({
            'Title': faker.sentence(nb_words=6),
            'Abstract': faker.paragraph(nb_sentences=5),
            'Classification Codes': faker.random_choices(elements=codes, length=2),
            'Date of Publication': faker.date_this_decade(),
            'Affiliations': faker.company(),
            'Citations': faker.random_int(min=0, max=100),
            'Keywords': faker.words(nb=5)
        })

    df = pd.DataFrame(data)
    classes = df['Classification Codes']
    return df, classes

# Load data
df, classes = load_mock_data() if use_mock else load_mock_data()

# Extract unique classification codes
all_codes = [code for sublist in df['Classification Codes'] for code in sublist]
unique_codes = sorted(set(all_codes))

# Initialize co-occurrence matrix
co_occurrence_matrix = np.zeros((len(unique_codes), len(unique_codes)), dtype=int)

# Fill the co-occurrence matrix
code_index = {code: idx for idx, code in enumerate(unique_codes)}
for codes in df['Classification Codes']:
    indices = [code_index[code] for code in codes]
    for i in indices:
        for j in indices:
            co_occurrence_matrix[i, j] += 1

co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=unique_codes, columns=unique_codes)

def main():

    st.title("Paper Research's Classification UMAP Clustering Chart")
    
    st.header("Co-Occurrence Matrix of Classification Codes")

    fig_co = px.imshow(co_occurrence_df, title='Co-Occurrence Matrix of Classification Codes', height=800, width=2000)
    st.plotly_chart(fig_co)

    # Convert to sparse matrix for efficiency
    sparse_matrix = csr_matrix(co_occurrence_matrix)

    # Use PCA to reduce dimensionality
    pca = PCA(n_components=50)
    pca_reduced = pca.fit_transform(sparse_matrix.toarray())

    # Reduce with UMAP
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # Apply k-means clustering on reduced data
    kmeans = KMeans(n_clusters=20, random_state=42)

    umap_embeddings = umap_reducer.fit_transform(pca_reduced)
    labels = kmeans.fit_predict(umap_embeddings)

    # Prepare DataFrame for visualization
    umap_df = pd.DataFrame({
        'UMAP Dimension 1': umap_embeddings[:, 0],
        'UMAP Dimension 2': umap_embeddings[:, 1],
        'Cluster': labels,
        'Code': unique_codes
    })

    # Scatter plot with hover labels
    fig_scatter = px.scatter(
        umap_df, 
        x='UMAP Dimension 1', 
        y='UMAP Dimension 2', 
        color='Cluster', 
        hover_data=['Code'],  # Add code names as hover labels
        title='UMAP Clustering of Classification Codes',
        labels={'UMAP Dimension 1': 'UMAP Dimension 1', 
                'UMAP Dimension 2': 'UMAP Dimension 2'}
    )

    col1, col2 = st.columns([5,5])
    with col1:
        st.plotly_chart(fig_scatter)
    # with col2:
        # st.write("Hi")

if __name__ == "__main__":
    main()
