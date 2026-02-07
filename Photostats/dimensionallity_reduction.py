from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def reduce_features(feature_vector, reduction_technique, n_components=2):
    """
    Reduces the dimensionality of the feature_vector using the specified reduction technique.

    Parameters:
        feature_vector (array-like): The input feature vectors.
        reduction_technique (str): The dimensionality reduction technique to use.
                                   Options: 'TSNE', 'PCA', 'UMAP', 'Diffusion_Map'
        n_components (int): Number of dimensions for the reduced space (default=2)

    Returns:
        np.ndarray: The reduced feature space with n_components.
    """
    technique = reduction_technique.upper()

    if technique == 'TSNE':
        reducer = TSNE(
            n_components=n_components,
            perplexity=10,
            learning_rate="auto",
            init="pca",
            random_state=42
        )
        reduced_space = reducer.fit_transform(feature_vector)

    elif technique == 'PCA':
        reducer = PCA(
            n_components=n_components,
            random_state=42
        )
        reduced_space = reducer.fit_transform(feature_vector)

    elif technique == 'UMAP':
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=10,
            min_dist=0.1,
            random_state=42
        )
        reduced_space = reducer.fit_transform(feature_vector)

 
    else:
        raise ValueError(f"Unknown reduction technique: {reduction_technique}")

    print(f"Reduced space shape: {reduced_space.shape}")
    return reduced_space
