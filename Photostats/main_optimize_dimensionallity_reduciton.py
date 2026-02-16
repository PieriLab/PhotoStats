import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from process_geometries import (
    GeometryDataset,
    generate_SOAP,
    generate_inv_eigenval,
    generate_inverse_dist_matrix,
    generate_MBTR,
    flatten_cartesian
)
from dimensionallity_reduction import reduce_features
from feature_comparison import full_embedding_analysis

# --------------------------
# Plotting functions
# --------------------------

def plot_line_for_fixed(hyperparam_values, metric_dict, metric_names,
                        sweep_name, fixed_val, fixed_name,
                        feature_name, reduction_method, save_folder="lineplots"):
    """
    Plot a line plot sweeping one hyperparameter for a fixed other hyperparameter.
    Each metric is a separate line.
    """
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(7,5))

    for metric_name in metric_names:
        y_vals = [metric_dict[h_val][fixed_val][metric_name] for h_val in hyperparam_values]
        plt.plot(hyperparam_values, y_vals, marker='o', label=metric_name)

    plt.xlabel(sweep_name)
    plt.ylabel("Metric Value")
    plt.title(f"{feature_name} - {reduction_method} | {fixed_name}={fixed_val}")
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(save_folder, f"{feature_name}_{reduction_method}_{fixed_name}_{fixed_val}_sweep.png")
    svg_path = os.path.join(save_folder, f"{feature_name}_{reduction_method}_{fixed_name}_{fixed_val}_sweep.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()


def plot_hyperparam_heatmap(score_matrix, hyperparam1_values, hyperparam2_values,
                            feature_name, metric_name, hyperparam_names=("Hyperparam1","Hyperparam2"),
                            save_folder="heatmaps"):
    """
    Plot 2D heatmap of metric scores.
    """
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(score_matrix, xticklabels=hyperparam2_values, yticklabels=hyperparam1_values,
                annot=True, fmt=".3f", cmap="viridis")
    plt.xlabel(hyperparam_names[1])
    plt.ylabel(hyperparam_names[0])
    plt.title(f"{feature_name} - {metric_name} heatmap")
    plt.tight_layout()

    png_path = os.path.join(save_folder, f"{feature_name}_{metric_name}_heatmap.png")
    svg_path = os.path.join(save_folder, f"{feature_name}_{metric_name}_heatmap.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()


# --------------------------
# Main function
# --------------------------

def main():
    # --------------------------
    # User parameters
    # --------------------------
    feature_name = "SOAP"        # <-- set the feature you want to analyze
    reduction_method = "UMAP"
    target_dim = 2

    hyperparam1_name = "n_neighbors"
    hyperparam2_name = "min_dist"
    hyperparam1_values = [5, 10, 20, 30]
    hyperparam2_values = [0.1, 0.5, 0.9]

    metrics = ['linear_accuracy', 'trustworthiness', 'silhouette', 'pearson_dist_corr']

    # Paths
    xyz_folder = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/aligned_geometries/seamstress_reflection/benzene/1_2/spawns"
    meci_labels_csv = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/meci_classification/benzene/S1S0/meci_labels_0.05_threshold_humanlabels.csv"
    output_dir = "embedding_hyperparam_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = GeometryDataset(xyz_folder, meci_labels_csv=meci_labels_csv)
    y = np.array(dataset.meci_labels)
    labeled_positions = np.arange(len(y))

    # Generate selected feature matrix
    if feature_name not in ["SOAP","inv_eigenval","inverse_dist_matrix","MBTR","flatten_cartesian"]:
        raise ValueError(f"Unknown feature: {feature_name}")

    feature_list = []
    for atoms, name in zip(dataset.structures, dataset.names):
        feature_list.append({
            "SOAP": generate_SOAP(atoms),
            "inv_eigenval": generate_inv_eigenval(atoms),
            "inverse_dist_matrix": generate_inverse_dist_matrix(atoms),
            "MBTR": generate_MBTR(atoms),
            "flatten_cartesian": flatten_cartesian(atoms),
        })

    X = np.vstack([f[feature_name] for f in feature_list])

    # --------------------------
    # Compute all metrics on grid
    # --------------------------
    results_grid = {h1:{h2:{} for h2 in hyperparam2_values} for h1 in hyperparam1_values}

    for h1 in hyperparam1_values:
        for h2 in hyperparam2_values:
            try:
                X_reduced = reduce_features(
                    feature_vector=X,
                    reduction_technique=reduction_method,
                    n_components=target_dim,
                    hyperparam1=h1,
                    hyperparam2=h2
                )
                metrics_result = full_embedding_analysis(
                    X_high=X,
                    X_low=X_reduced,
                    labeled_positions=labeled_positions,
                    y_labeled=y
                )
                results_grid[h1][h2] = metrics_result

            except Exception as e:
                print(f"Failed: {feature_name}, {reduction_method}, {hyperparam1_name}={h1}, {hyperparam2_name}={h2} → {e}")
                results_grid[h1][h2] = {m: np.nan for m in metrics}

    # --------------------------
    # Line plots: sweep hyperparam1 for each fixed hyperparam2
    # --------------------------
    lineplot_folder = os.path.join(output_dir,"lineplots")
    for fixed_h2 in hyperparam2_values:
        plot_line_for_fixed(hyperparam1_values, results_grid, metrics,
                            sweep_name=hyperparam1_name, fixed_val=fixed_h2, fixed_name=hyperparam2_name,
                            feature_name=feature_name, reduction_method=reduction_method,
                            save_folder=lineplot_folder)

    # Sweep hyperparam2 for each fixed hyperparam1
    for fixed_h1 in hyperparam1_values:
        # Need to transpose dict for this sweep
        results_grid_transpose = {h2:{h1:{} for h1 in hyperparam1_values} for h2 in hyperparam2_values}
        for h1 in hyperparam1_values:
            for h2 in hyperparam2_values:
                results_grid_transpose[h2][h1] = results_grid[h1][h2]

        plot_line_for_fixed(hyperparam2_values, results_grid_transpose, metrics,
                            sweep_name=hyperparam2_name, fixed_val=fixed_h1, fixed_name=hyperparam1_name,
                            feature_name=feature_name, reduction_method=reduction_method,
                            save_folder=lineplot_folder)

    # --------------------------
    # Heatmaps
    # --------------------------
    heatmap_folder = os.path.join(output_dir,"heatmaps")
    for m in metrics:
        score_matrix = np.array([[results_grid[h1][h2][m] for h2 in hyperparam2_values] for h1 in hyperparam1_values])
        plot_hyperparam_heatmap(score_matrix, hyperparam1_values, hyperparam2_values,
                                feature_name, m,
                                hyperparam_names=(hyperparam1_name, hyperparam2_name),
                                save_folder=heatmap_folder)


if __name__ == "__main__":
    main()
