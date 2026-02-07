from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
import numpy as np


def full_embedding_analysis(
    X_high,
    X_low,
    labeled_positions,
    y_labeled
):
    """
    Run full analysis on a reduced embedding.
    """
    results = {}

    # linear classifier accuracy
    results["linear_accuracy"] = linear_classifier_accuracy(
        reduced_feature=X_low,
        labeled_positions=labeled_positions,
        y_labeled=y_labeled
    )

    # trustworthiness
    results["trustworthiness"] = trustworthiness_score(
        X_high,
        X_low
    )

    # silhouette score
    results["silhouette"] = silhouette_score_safe(
        X_low,
        y_labeled
    )

    # pearson distance correlation
    results["pearson_dist_corr"] = pearson_distance_correlation(
        X_high,
        X_low
    )

    return results


def linear_classifier_accuracy(
    reduced_feature,
    labeled_positions,
    y_labeled,
    test_size=0.2,
    random_state=42
):
    """
    Train a linear classifier on labeled points and return test accuracy.
    """
    # Need at least two classes
    if len(set(y_labeled)) <= 1:
        return None

    X = reduced_feature[labeled_positions]
    y = np.array(y_labeled)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs"
    )
    clf.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    return acc

def trustworthiness_score(X_high, X_low, n_neighbors=10):
    return trustworthiness(
        X_high,
        X_low,
        n_neighbors=n_neighbors
    )

def silhouette_score_safe(X_low, labels):
    if len(set(labels)) <= 1:
        return None
    return silhouette_score(X_low, labels)


def pearson_distance_correlation(X_high, X_low, max_points=1000):
    """
    Pearson correlation between pairwise distances in high- and low-D spaces.
    Subsamples points for efficiency.
    """
    n = X_high.shape[0]

    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]

    D_high = pairwise_distances(X_high)
    D_low = pairwise_distances(X_low)

    iu = np.triu_indices_from(D_high, k=1)
    return pearsonr(D_high[iu], D_low[iu])[0]

