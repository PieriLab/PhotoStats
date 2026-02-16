from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def reduce_features(feature_vector, reduction_technique, n_components=2,
                    hyperparam1=None, hyperparam2=None):
    """
    Reduces the dimensionality of the feature_vector using the specified reduction technique
    and allows sweeping of two hyperparameters.

    Parameters:
        feature_vector (array-like): The input feature vectors.
        reduction_technique (str): Dimensionality reduction technique ('TSNE', 'PCA', 'UMAP').
        n_components (int): Number of dimensions for reduced space (default=2)
        hyperparam1: First hyperparameter (e.g., n_neighbors or perplexity)
        hyperparam2: Second hyperparameter (e.g., min_dist or learning_rate)

    Returns:
        np.ndarray: Reduced feature space with shape (n_samples, n_components)
    """
    technique = reduction_technique.upper()

    if technique == 'TSNE':
        reducer = TSNE(
            n_components=n_components,
            perplexity=hyperparam1 if hyperparam1 is not None else 10,
            learning_rate=hyperparam2 if hyperparam2 is not None else "auto",
            init="pca",
            random_state=42
        )
        reduced_space = reducer.fit_transform(feature_vector)

    elif technique == 'PCA':
        # PCA doesn’t have many hyperparameters, can ignore for now
        reducer = PCA(
            n_components=n_components,
            random_state=42
        )
        reduced_space = reducer.fit_transform(feature_vector)

    elif technique == 'UMAP':
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=hyperparam1 if hyperparam1 is not None else 10,
            min_dist=hyperparam2 if hyperparam2 is not None else 0.1,
            random_state=42
        )
        reduced_space = reducer.fit_transform(feature_vector)

    else:
        raise ValueError(f"Unknown reduction technique: {reduction_technique}")

    return reduced_space
