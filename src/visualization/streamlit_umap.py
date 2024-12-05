import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import plotly.express as px
import plotly.graph_objects as go
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
    codes = []
    for i in range (100):
        codes.append('Class{0}'.format(i))
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

df = None
classes = None

if use_mock:
    df, classes = load_mock_data()
else:
    # in reality need to use some func but now don't know so we will use load_mock too
    df, classes = load_mock_data()

all_codes = [code for sublist in df['Classification Codes'] for code in sublist]
unique_codes = sorted(list(set(all_codes)))

co_occurrence_matrix = np.zeros((len(unique_codes), len(unique_codes)), dtype=int)

# Fill the co-occurrence matrix
for codes in df['Classification Codes']:
    for i, code1 in enumerate(unique_codes):
        for j, code2 in enumerate(unique_codes):
            if code1 in codes and code2 in codes:
                co_occurrence_matrix[i, j] += 1

co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=unique_codes, columns=unique_codes)

def main():
    # st.set_page_config(page_title="Paper Research's Classification UMAP Clustering Chart", layout="wide")
    
    st.title("Paper Research's Classification UMAP Clustering Chart")
    
    # st.sidebar()

    st.header("Co-Occurrence Matrix of Classification Codes")

    fig_co = px.imshow(co_occurrence_df)

    fig_co.update_layout(
        title='Feature Relations by Species',
        height=800
        # margin=dict(l=10, r=10, b=10, t=10, pad=10),
    )

    st.plotly_chart(fig_co, use_container_width=True)

    # Convert to sparse matrix for efficiency
    sparse_matrix = csr_matrix(co_occurrence_matrix)

    # Use PCA to reduce dimensionality
    pca = PCA(n_components=50)
    pca_reduced = pca.fit_transform(sparse_matrix.toarray())

    # Reduce with UMAP
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(pca_reduced)

    # Apply k-means clustering on reduced data
    kmeans = KMeans(n_clusters=20, random_state=42)
    labels = kmeans.fit_predict(umap_embeddings)

    fig_scatter = px.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1])

    fig_scatter.update_layout(
        title='UMAP',
        height=800
        # margin=dict(l=10, r=10, b=10, t=10, pad=10),
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # umap_2d = UMAP(random_state=0)

    # umap_2d.fit()

if __name__ == "__main__":
    main()