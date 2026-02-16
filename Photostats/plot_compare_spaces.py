import plotly.graph_objects as go
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def plot_3d_embedding_plotly(
    X_3d,
    labels,
    is_meci_mask,
    meci_names,
    feature_name,
    reduction_method,
    class_colors,
    meci_star_colors,
    save_folder="plots_html"
):

    os.makedirs(save_folder, exist_ok=True)

    fig = go.Figure()

    unique_classes = sorted(np.unique(labels))

    # ------------------------
    # Non-MECI geometries
    # ------------------------
    for i, cls in enumerate(unique_classes):
        cls_mask = (labels == cls) & (~is_meci_mask)
        color = class_colors[i % len(class_colors)]

        fig.add_trace(
            go.Scatter3d(
                x=X_3d[cls_mask, 0],
                y=X_3d[cls_mask, 1],
                z=X_3d[cls_mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
                    opacity=0.8
                ),
                name=f"Class {cls}"
            )
        )

    # ------------------------
    # MECIs (stars with names)
    # ------------------------
    meci_indices = np.where(is_meci_mask)[0]

    for i, (idx, name) in enumerate(zip(meci_indices, meci_names)):
        color = meci_star_colors[i % len(meci_star_colors)]

        fig.add_trace(
            go.Scatter3d(
                x=[X_3d[idx, 0]],
                y=[X_3d[idx, 1]],
                z=[X_3d[idx, 2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                    symbol="diamond",
                    line=dict(width=1, color="black")
                ),
                name=name,
                hovertext=name,
                hoverinfo="text"
            )
        )

    fig.update_layout(
        title=f"{feature_name} — {reduction_method} (3D)",
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3"
        )
    )

    filename = f"{save_folder}/{feature_name}_{reduction_method}_3D.html"
    fig.write_html(filename)


def plot_2d_embedding(
    X_2d,
    labels,
    is_meci_mask,
    meci_names,
    feature_name,
    reduction_method,
    class_colors,
    meci_star_colors,
    save_folder="plots",
    save_svg=True,
    save_png=True
):
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(8, 6))

    unique_classes = sorted(np.unique(labels))

    # Non-MECI
    for i, cls in enumerate(unique_classes):
        cls_mask = (labels == cls) & (~is_meci_mask)
        color = class_colors[i % len(class_colors)]

        plt.scatter(
            X_2d[cls_mask, 0],
            X_2d[cls_mask, 1],
            c=color,
            label=f"Class {cls}",
            alpha=0.7,
            s=40
        )

    # MECIs with names
    meci_indices = np.where(is_meci_mask)[0]

    for i, (idx, name) in enumerate(zip(meci_indices, meci_names)):
        color = meci_star_colors[i % len(meci_star_colors)]

        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            marker="*",
            c=color,
            s=250,
            edgecolor="k",
            linewidth=0.8,
            label=name
        )

    plt.title(f"{feature_name} — {reduction_method}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(fontsize=8)
    plt.tight_layout()

    base_filename = f"{save_folder}/{feature_name}_{reduction_method}_2D"

    if save_png:
        plt.savefig(base_filename + ".png", dpi=300)

    if save_svg:
        plt.savefig(base_filename + ".svg")

    plt.close()


import seaborn as sns

def plot_metrics_heatmap(df, metric, save_folder="plots", save_png=True, save_svg=True):
    os.makedirs(save_folder, exist_ok=True)
    
    # Create a column combining feature and reduction
    df['feature_reduction'] = df['feature'] + '-' + df['reduction']
    
    # Pivot: rows=feature_reduction, columns=n_components, values=metric
    heatmap_data = df.pivot(index='feature_reduction', columns='n_components', values=metric)
    
    plt.figure(figsize=(10, max(6, 0.5*len(heatmap_data))))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", cbar=True)
    plt.title(f"{metric} Across Features, Reductions, and Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Feature-Reduction")
    plt.tight_layout()
    
    if save_png:
        path_png = os.path.join(save_folder, f"{metric}_heatmap.png")
        plt.savefig(path_png, dpi=300)
    if save_svg:
        path_svg = os.path.join(save_folder, f"{metric}_heatmap.svg")
        plt.savefig(path_svg)
    
    #plt.show()
    #plt.close()


from collections import Counter

def plot_meci_class_distribution(dataset, save_folder="plots", class_colors=None, save_png=True, save_svg=True):
    """
    Pie chart showing MECI class distribution.
    """
    import matplotlib.pyplot as plt
    import os
    from collections import Counter

    os.makedirs(save_folder, exist_ok=True)

    # Count MECI classes
    counts = Counter(dataset.meci_labels)
    labels = list(counts.keys())
    values = list(counts.values())

    # Use provided colors if given
    if class_colors is None:
        class_colors = plt.cm.tab10.colors[:len(labels)]
    else:
        class_colors = class_colors[:len(labels)]

    plt.figure(figsize=(6,6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=class_colors, startangle=90)
    plt.title("MECI Class Distribution")
    plt.axis('equal')  # Make it circular

    if save_png:
        plt.savefig(os.path.join(save_folder, "meci_class_distribution.png"), dpi=300)
    if save_svg:
        plt.savefig(os.path.join(save_folder, "meci_class_distribution.svg"))

    #plt.show()
    #plt.close()



from scipy.spatial.distance import pdist, squareform



def plot_violin_inter_intra(
    X,
    labels,
    save_folder,
    feature_name,
    reduction_method=None,
    dim=None,
    save_png=True,
    save_svg=True
):
    """
    Plot a single violin showing inter-class distances on left and intra-class distances on right.
    Can handle the case where no dimensionality reduction was applied (dim=None, reduction_method=None).
    """

    os.makedirs(save_folder, exist_ok=True)
    labels = np.array(labels)
    n = X.shape[0]

    # Compute pairwise distances
    D = squareform(pdist(X))

    intra_distances = []
    inter_distances = []

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_distances.append(D[i, j])
            else:
                inter_distances.append(D[i, j])

    df_plot = pd.DataFrame({
        "distance": inter_distances + intra_distances,
        "type": ["inter"]*len(inter_distances) + ["intra"]*len(intra_distances)
    })

    plt.figure(figsize=(6,6))
    # Force inter on left, intra on right
    sns.violinplot(x="type", y="distance", data=df_plot, palette=["#ff7f0e", "#1f77b4"], order=["inter","intra"])

    # Handle titles for no reduction
    title_reduction = reduction_method if reduction_method is not None else "NONE"
    title_dim = f"{dim}D" if dim is not None else "original"

    plt.title(f"{feature_name} - {title_reduction} ({title_dim}) Distances")
    plt.ylabel("Pairwise Distance")
    plt.xlabel("")
    plt.tight_layout()

    # Filenames
    filename_base = f"violin_{feature_name}_{title_reduction}_{title_dim}_distances"
    if save_png:
        plt.savefig(os.path.join(save_folder, f"{filename_base}.png"), dpi=300)
    if save_svg:
        plt.savefig(os.path.join(save_folder, f"{filename_base}.svg"))

    #plt.show()
    #plt.close()


def plot_violin_inter_and_intra_per_class(
    X,
    labels,
    save_folder,
    feature_name,
    reduction_method=None,
    dim=None,
    save_png=True,
    save_svg=True
):
    """
    Plot:
    - One violin for all inter-class distances
    - One violin per class for intra-class distances
    """

    os.makedirs(save_folder, exist_ok=True)

    labels = np.array(labels)
    unique_classes = np.unique(labels)
    n = X.shape[0]

    # Compute pairwise distances
    D = squareform(pdist(X))

    inter_distances = []
    intra_by_class = {cls: [] for cls in unique_classes}

    # Collect distances
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_by_class[labels[i]].append(D[i, j])
            else:
                inter_distances.append(D[i, j])

    # Build plotting dataframe
    data = []
    
    # Inter-class (single violin)
    for d in inter_distances:
        data.append(("inter", d))

    # Intra-class (one violin per class)
    for cls in unique_classes:
        for d in intra_by_class[cls]:
            data.append((f"intra_{cls}", d))

    df_plot = pd.DataFrame(data, columns=["type", "distance"])

    # Order: inter first, then each intra class
    order = ["inter"] + [f"intra_{cls}" for cls in unique_classes]

    plt.figure(figsize=(8, 6))
    sns.violinplot(
        x="type",
        y="distance",
        data=df_plot,
        order=order,
        inner="box"
    )

    # Title handling
    title_reduction = reduction_method if reduction_method is not None else "NONE"
    title_dim = f"{dim}D" if dim is not None else "original"

    plt.title(f"{feature_name} - {title_reduction} ({title_dim}) Distances")
    plt.ylabel("Pairwise Distance")
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename_base = f"violin_{feature_name}_{title_reduction}_{title_dim}_inter_intra_per_class"

    if save_png:
        plt.savefig(os.path.join(save_folder, f"{filename_base}.png"), dpi=300)
    if save_svg:
        plt.savefig(os.path.join(save_folder, f"{filename_base}.svg"))

    # plt.show()
    # plt.close()