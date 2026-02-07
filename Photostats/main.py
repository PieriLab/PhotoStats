import argparse
import numpy as np
import pandas as pd 
from tqdm import tqdm
from dimensionallity_reduction import reduce_features
from feature_comparison import full_embedding_analysis, linear_classifier_accuracy
from process_geometries import (
    GeometryDataset,
    generate_SOAP,
    generate_inv_eigenval,
    generate_inverse_dist_matrix,
    generate_MBTR,
    flatten_cartesian
)

def results_dict_to_df(results):
    rows = []

    for feature, methods in results.items():
        for method, dims in methods.items():
            for dim, metrics in dims.items():

                # Handle NONE case
                if dim == "full":
                    n_dim = None
                else:
                    n_dim = dim

                row = {
                    "feature": feature,
                    "reduction": method,
                    "n_components": n_dim,
                }

                # Add all metrics
                for k, v in metrics.items():
                    row[k] = v

                rows.append(row)

    return pd.DataFrame(rows)

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
    print(dataset.names, dataset.meci_labels)
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


                metrics = linear_classifier_accuracy(
                    reduced_feature=X,
                    labeled_positions=labeled_positions,
                    y_labeled=y
                )
                

                if isinstance(metrics, float):
                    metrics = {"linear_accuracy": metrics}

                results[feature_name][method]["full"] = metrics
                continue

            # check all dimensions
            for dim in target_dims:

            # no highdim tsne
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

                    metrics = full_embedding_analysis(
                                X_high=X,
                                X_low=X_reduced,
                                labeled_positions=labeled_positions,
                                y_labeled=y
                                )

                except Exception as e:
                    print(f"Failed: {feature_name}, {method}, dim={dim} → {e}")
                    acc = None

                results[feature_name][method][dim] = metrics

    print(results)

    results_df = results_dict_to_df(results)
    output_path = "embedding_analysis_results.csv"
    results_df.to_csv(output_path, index=False)



        

if __name__ == "__main__":
    main()
