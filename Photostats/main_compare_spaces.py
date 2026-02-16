import argparse
from collections import Counter
import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
from dimensionallity_reduction import reduce_features
from feature_comparison import full_embedding_analysis, linear_classifier_accuracy
from plot_compare_spaces import plot_2d_embedding, plot_3d_embedding_plotly, plot_metrics_heatmap,plot_meci_class_distribution, plot_violin_inter_intra, plot_violin_inter_and_intra_per_class
from process_geometries import (
    GeometryDataset,
    generate_SOAP,
    generate_inv_eigenval,
    generate_inverse_dist_matrix,
    generate_MBTR,
    flatten_cartesian
)

def count_meci_classes(self):
    if self.meci_labels is None:
        raise RuntimeError("Dataset was created without MECI labels.")

    return Counter(self.meci_labels)
    
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

CLASS_COLORS = [
    "#000000",  # class 0

    "#1f77b4",  # class 0
    "#ff7f0e",  # class 1
    "#2ca02c",  # class 2
    "#d62728",
    "#9467bd",
    "#8c564b",
]

MECI_STAR_COLORS = [
    "#1f77b4", 
    "#ff7f0e",  
    "#2ca02c",  
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#c25ba8",
    "#35d4c2",
    "#000000",
]



def main():

  

    xyz_folder = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/aligned_geometries/seamstress_reflection/benzene/1_2/spawns"

    meci_folder = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/aligned_geometries/seamstress_reflection/benzene/1_2/mecis"

    meci_labels = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/meci_classification/benzene/S1S0/meci_labels_0.05_threshold_humanlabels.csv"

    output_dir = "embedding_outputs_benzene"
    os.makedirs(output_dir, exist_ok=True)

    plot2d_dir = os.path.join(output_dir, "plots_2D")
    plot3d_dir = os.path.join(output_dir, "plots_3D")
    os.makedirs(plot2d_dir, exist_ok=True)
    os.makedirs(plot3d_dir, exist_ok=True)



    dataset = GeometryDataset(
        folder_path=xyz_folder,
        meci_labels_csv=meci_labels
    )

    print(f"Loaded {len(dataset)} labeled structures\n")


    plot_meci_class_distribution(
        dataset,
        class_colors=CLASS_COLORS,
        save_folder=os.path.join(output_dir, "plots", "meci_distribution")
    )




    meci_dataset = GeometryDataset(
        folder_path=meci_folder,
        meci_labels_csv=None
    )

    print(f"Loaded {len(meci_dataset)} MECI geometries\n")

    # GENERATE FEATURES


    feature_list = []
    for atoms, name in tqdm(
        zip(dataset.structures, dataset.names),
        total=len(dataset),
        desc="Generating labeled features"
    ):
        feature_list.append({
            "name": name,
            "SOAP": generate_SOAP(atoms),
            "inv_eigenval": generate_inv_eigenval(atoms),
            "inverse_dist_matrix": generate_inverse_dist_matrix(atoms),
            "MBTR": generate_MBTR(atoms),
            "flatten_cartesian": flatten_cartesian(atoms),
        })

    # MECI feature list
    meci_feature_list = []
    for atoms, name in tqdm(
        zip(meci_dataset.structures, meci_dataset.names),
        total=len(meci_dataset),
        desc="Generating MECI features"
    ):
        meci_feature_list.append({
            "name": name,
            "SOAP": generate_SOAP(atoms),
            "inv_eigenval": generate_inv_eigenval(atoms),
            "inverse_dist_matrix": generate_inverse_dist_matrix(atoms),
            "MBTR": generate_MBTR(atoms),
            "flatten_cartesian": flatten_cartesian(atoms),
        })


    feature_names = ["SOAP", "inv_eigenval", "inverse_dist_matrix", "MBTR", "flatten_cartesian"]

    feature_matrices = {
        fname: np.vstack([f[fname] for f in feature_list])
        for fname in feature_names
    }

    meci_feature_matrices = {
        fname: np.vstack([f[fname] for f in meci_feature_list])
        for fname in feature_names
    }

    y = np.array(dataset.meci_labels)
    labeled_positions = np.arange(len(y))

    reduction_methods = ["NONE", "PCA", "UMAP", "TSNE"]
    target_dims = [1,2,3,4,5]

    results = {}



    for feature_name in feature_names:

        print(f"\n=== Feature: {feature_name} ===")

        X_no_meci = feature_matrices[feature_name]
        X_meci = meci_feature_matrices[feature_name]

        # concatenate for visualization
        X_with_meci = np.vstack([X_no_meci, X_meci])

        results[feature_name] = {}

        for method in reduction_methods:

            results[feature_name][method] = {}

            # NO REDUCTION (ANALYSIS ONLY)
            if method == "NONE":

                metrics = linear_classifier_accuracy(
                    reduced_feature=X_no_meci,
                    labeled_positions=labeled_positions,
                    y_labeled=y
                )
                plot_violin_inter_intra(
                    X=X_no_meci,
                    labels=y,
                    save_folder=os.path.join(output_dir, "plots", "violin_plots"),
                    feature_name=feature_name,
                )

                plot_violin_inter_and_intra_per_class(
                    X=X_no_meci,
                    labels=y,
                    save_folder=os.path.join(output_dir, "plots", "violin_plots"),
                    feature_name=feature_name,
                )


                if isinstance(metrics, float):
                    metrics = {"linear_accuracy": metrics}

                results[feature_name][method]["full"] = metrics
                continue


     
            # DIMENSION LOOP
            for dim in target_dims:

                if dim >= X_no_meci.shape[1]:
                    continue

                try:

           
                    X_reduced_no_meci = reduce_features(
                        feature_vector=X_no_meci,
                        reduction_technique=method,
                        n_components=dim
                    )

                    metrics = full_embedding_analysis(
                        X_high=X_no_meci,
                        X_low=X_reduced_no_meci,
                        labeled_positions=labeled_positions,
                        y_labeled=y
                    )

                    results[feature_name][method][dim] = metrics

                    # Violin plot of inter vs intra distances
                    plot_violin_inter_intra(
                        X=X_reduced_no_meci,
                        labels=y,
                        save_folder=os.path.join(output_dir, "plots", "violin_plots"),
                        feature_name=feature_name,
                        reduction_method=method,
                        dim=dim
                    )



                 


                    # PLOTTING
                    if dim == 2 or dim == 3:

                        X_reduced_with_meci = reduce_features(
                            feature_vector=X_with_meci,
                            reduction_technique=method,
                            n_components=dim
                        )

                        n_non_meci = X_no_meci.shape[0]
                        n_total = X_with_meci.shape[0]

                        is_meci_mask = np.zeros(n_total, dtype=bool)
                        is_meci_mask[n_non_meci:] = True

                        # ---- 2D PLOTS ----
                        if dim == 2:
                            plot_2d_embedding(
                            X_2d=X_reduced_with_meci,
                            labels=np.concatenate([y, np.zeros(n_total - n_non_meci)]),
                            is_meci_mask=is_meci_mask,
                            meci_names=meci_dataset.names, 
                            feature_name=feature_name,
                            reduction_method=method,
                            class_colors=CLASS_COLORS,
                            meci_star_colors=MECI_STAR_COLORS,
                            save_folder=plot2d_dir,
                            save_svg=True,
                            save_png=True
)


                        # ---- 3D PLOTS ----
                        if dim == 3:
                            plot_3d_embedding_plotly(
                                X_3d=X_reduced_with_meci,
                                labels=np.concatenate([y, np.zeros(n_total - n_non_meci)]),
                                is_meci_mask=is_meci_mask,
                                meci_names=meci_dataset.names, 
                                feature_name=feature_name,
                                reduction_method=method,
                                class_colors=CLASS_COLORS,
                                meci_star_colors=MECI_STAR_COLORS,
                                save_folder=plot3d_dir
                            )

                except Exception as e:
                    print(f"Failed: {feature_name}, {method}, dim={dim} → {e}")

  

    results_df = results_dict_to_df(results)

    csv_path = os.path.join(output_dir, "embedding_analysis_results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\nSaved results to: {csv_path}")

    metrics_to_plot = ['linear_accuracy', 'trustworthiness', 'silhouette', 'pearson_dist_corr']

    heatmap_folder = os.path.join(output_dir, "plots", "heatmaps")
    os.makedirs(heatmap_folder, exist_ok=True)

    for metric in metrics_to_plot:
        plot_metrics_heatmap(results_df, metric, save_folder=heatmap_folder)


if __name__ == "__main__":



    main()

    print("\nFinished successfully.\n")
