from Photostats.dim_red import reduce_features
from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt


def compare_dim_reduction(feature_vector, max_dimension=4):
    reduction_techniques = ['UMAP', 'TSNE', 'PCA']
    results = {}

    for red_technique in reduction_techniques:
        dim_trustworthiness = []

        for dim in range(1, max_dimension + 1):
            reduced_feature = reduce_features(
                feature_vector,
                reduction_technique=red_technique,
                n_components=dim
            )

            score = trustworthiness(feature_vector, reduced_feature)
            dim_trustworthiness.append(score)

        results[red_technique] = dim_trustworthiness

    return results
