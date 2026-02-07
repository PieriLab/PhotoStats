import argparse
import numpy as np
from tqdm import tqdm
from dimensionallity_reduction import reduce_features
from feature_comparison import linear_classifier_accuracy
from process_geometries import (
    GeometryDataset,
    generate_SOAP,
    generate_inv_eigenval,
    generate_inverse_dist_matrix,
    generate_MBTR,
    flatten_cartesian
)

def main():
    #parser = argparse.ArgumentParser(description="Process XYZ geometries and MECI labels")
    #parser.add_argument("folder", type=str, help="Path to folder containing XYZ files")
    #parser.add_argument("meci_class_csv", type=str, help="Path to CSV of MECI labels")

    #args = parser.parse_args()


    ## load in data
    xyz_folder = '/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/aligned_geometries/SeamStress/benzene/type2/spawn'

    



    meci_labels = '/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/meci_classification/benzene/S1S0/meci_labels_humanlabels.csv'
   
    #generate feature sets

    dataset = GeometryDataset(
    folder_path=xyz_folder,
    meci_labels_csv=meci_labels
    )

    print(f"Loaded {len(dataset)} structures:\n")

    feature_list = []

    for atoms, name in tqdm(
        zip(dataset.structures, dataset.names),
        total=len(dataset),
        desc="Generating features"
        ):

        features = {
            "name": name,
            "SOAP": generate_SOAP(atoms),
            "inv_eigenval": generate_inv_eigenval(atoms),
            "inverse_dist_matrix": generate_inverse_dist_matrix(atoms),
            "MBTR": generate_MBTR(atoms),
            "flatten_cartesian": flatten_cartesian(atoms),
        }
        feature_list.append(features)


    

    y = np.array(dataset.meci_labels)
    labeled_positions = np.arange(len(y))  

    # Build feature matrices
    feature_matrices = {
        "SOAP": np.vstack([f["SOAP"] for f in feature_list]),
        "inv_eigenval": np.vstack([f["inv_eigenval"] for f in feature_list]),
        "inverse_dist_matrix": np.vstack([f["inverse_dist_matrix"] for f in feature_list]),
        "MBTR": np.vstack([f["MBTR"] for f in feature_list]),
        "flatten_cartesian": np.vstack([f["flatten_cartesian"] for f in feature_list]),
    }


    reduction_methods = ["NONE", "PCA", "UMAP", "TSNE"]

    target_dims = [1, 2, 3]

    results = {}

    for feature_name, X in feature_matrices.items():
        print(f"\n=== Feature: {feature_name} ===")
        results[feature_name] = {}

        for method in reduction_methods:
            results[feature_name][method] = {}

            if method == "NONE":
                acc = linear_classifier_accuracy(
                    reduced_feature=X,
                    labeled_positions=labeled_positions,
                    y_labeled=y
                )
                results[feature_name][method]["full"] = acc
                continue

        # Dimensionality r
            for dim in target_dims:

            # Practical guardrails
                if method == "TSNE" and dim > 3:
                    continue
                if dim >= X.shape[1]:
                    continue

                try:
                    X_reduced = reduce_features(
                        feature_vector=X,
                        reduction_technique=method,
                        n_components=dim
                    )

                    acc = linear_classifier_accuracy(
                        reduced_feature=X_reduced,
                        labeled_positions=labeled_positions,
                        y_labeled=y
                    )

                except Exception as e:
                    print(f"Failed: {feature_name}, {method}, dim={dim} → {e}")
                    acc = None

                results[feature_name][method][dim] = acc

    print(results)


        

if __name__ == "__main__":
    main()
