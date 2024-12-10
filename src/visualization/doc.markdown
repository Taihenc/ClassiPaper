# ClassiPaper: Paper Research Classification Prediction Clustering and Visualization

## Overview
ClassiPaper is a Streamlit application that provides clustering and visualization of research paper classifications. It uses UMAP for dimensionality reduction and KMeans for clustering. The application also includes error analysis between actual and predicted classifications.

## Features
- Load mock data or real data
- Preprocess data and generate embeddings
- Perform clustering and dimensionality reduction
- Visualize clusters in 3D
- Display class distribution
- Perform error analysis between actual and predicted classifications

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/classipaper.git
    cd classipaper
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Run the Streamlit application:
    ```sh
    streamlit run streamlit_umap.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

## Settings
### Sidebar
- **Class Distribution Settings**: Adjust the number of classes to display in the class distribution bar chart.
- **Show Error Analysis**: Toggle the display of error analysis between actual and predicted classifications.

## Data Loading
- The application uses mock data by default. To use real data, modify the `use_mock` variable to `False` and provide your data loading logic.

## Clustering and Visualization
- The application performs clustering and dimensionality reduction using UMAP and KMeans.
- Visualize the clusters in 3D scatter plots for both actual and predicted data.

## Error Analysis
- The application computes Jaccard similarity between actual and predicted classifications.
- View a summary of error metrics and a detailed error analysis table.

## Example
Here is an example of how to use the application with mock data:

1. Run the application:
    ```sh
    streamlit run ./src/visualization/dsde_viz.py
    ```

2. Adjust the settings in the sidebar as needed.

3. View the clustering visualization and class distribution.

4. Toggle the error analysis to view the error metrics and detailed analysis.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.