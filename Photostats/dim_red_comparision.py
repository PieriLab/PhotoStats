import pandas as pd
import numpy as np
from Photostats.dim_red import reduce_features
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def create_meci_df(idx_list, meci_labels_csv):
    """
    Creates a DataFrame aligned with feature_vector containing MECI labels.
    
    Parameters:
    - idx_list: list of identifiers corresponding to feature_vector rows
    - meci_labels_csv: path to CSV containing ['idx', 'meci_type']

    Returns:
    - labeled_df: pd.DataFrame with columns ['idx', 'meci_label'], ordered same as idx_list
                  Rows with no label get NaN in 'meci_label'
    """
    df = pd.read_csv(meci_labels_csv)
    meci_labels_dict = dict(zip(df['idx'], df['meci_type']))

    labels_aligned = [meci_labels_dict.get(id_, np.nan) for id_ in idx_list]

    labeled_df = pd.DataFrame({
        'idx': idx_list,
        'meci_label': labels_aligned
    })

    print(f"Total points: {len(idx_list)}, Labeled points: {labeled_df['meci_label'].notna().sum()}")

    return labeled_df


def compare_dimensionallity_reduction(feature_vector, labeled_df, max_dimension=4):
    """
    Perform dimensionality reduction, compute trustworthiness, and linear classification accuracy
    for labeled points.

    Parameters:
    - feature_vector: np.array of shape (num_points, n_features)
    - labeled_df: pd.DataFrame with columns ['idx', 'meci_label'] aligned with feature_vector
    - max_dimension: maximum reduced dimension to test

    Returns:
    - results: dict of the form:
        { technique: {'trustworthiness': [...], 'linear_accuracy': [...] } }
    """
    reduction_techniques = ['UMAP', 'TSNE', 'PCA']
    results = {}

    # Find indices of labeled points
    labeled_positions = labeled_df.index[labeled_df['meci_label'].notna()].tolist()
    y_labeled = labeled_df['meci_label'].iloc[labeled_positions].tolist()

    for red_technique in reduction_techniques:
        dim_trustworthiness = []
        dim_linear_accuracy = []

        for dim in range(1, max_dimension + 1):
            # 1. Reduce features
            reduced_feature = reduce_features(
                feature_vector,
                reduction_technique=red_technique,
                n_components=dim
            )

            # 2. Trustworthiness
            tw_score = trustworthiness(feature_vector, reduced_feature)
            dim_trustworthiness.append(tw_score)

            # 3. Linear classifier accuracy on labeled points
            if len(set(y_labeled)) > 1:  # at least 2 classes
                X = reduced_feature[labeled_positions]
                y = y_labeled

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
                clf.fit(X_train_scaled, y_train)

                acc = accuracy_score(y_test, clf.predict(X_test_scaled))
                dim_linear_accuracy.append(acc)
            else:
                dim_linear_accuracy.append(None)

        results[red_technique] = {
            'trustworthiness': dim_trustworthiness,
            'linear_accuracy': dim_linear_accuracy
        }

    return results
