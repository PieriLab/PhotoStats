import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd

def plot_spawns_vs_mecis(X_spawns, X_mecis, smiles_spawns, smiles_mecis, idx_mecis, reduction_technique="UMAP", style_file=None):
    """
    Scatter plot of spawns vs MECIs, colored by SMILES.
    """
    unique_smiles = list(dict.fromkeys(smiles_spawns + smiles_mecis))

    color_list = [
        "tab:red","tab:orange","tab:green","tab:blue","tab:purple",
        "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan",
    ]
    if len(unique_smiles) > len(color_list):
        raise ValueError("Not enough colors for number of unique SMILES")

    smiles_to_color = {smi: color_list[i] for i, smi in enumerate(unique_smiles)}

    if style_file:
        plt.style.use(style_file)

    fig, ax = plt.subplots(figsize=(6,6))

    for smi in unique_smiles:
        mask = [s == smi for s in smiles_spawns]
        ax.scatter(
            X_spawns[mask,0],
            X_spawns[mask,1],
            color=smiles_to_color[smi],
            s=20,
            alpha=0.6
        )

    for (x,y), smi in zip(X_mecis, smiles_mecis):
        ax.scatter(x, y, marker="*", s=200, color=smiles_to_color[smi], edgecolor="k", linewidth=0.8)

    smiles_handles = [mlines.Line2D([], [], marker='o', linestyle='', color=smiles_to_color[smi], label=smi, markersize=8) for smi in unique_smiles]
    legend_smiles = ax.legend(handles=smiles_handles, title="SMILES", loc="upper left", bbox_to_anchor=(0.97,1), fontsize=8)
    ax.add_artist(legend_smiles)

    meci_handles = [mlines.Line2D([], [], marker='*', linestyle='', color=smiles_to_color[smi], markeredgecolor='k', markeredgewidth=0.8, label=name, markersize=12) for name, smi in zip(idx_mecis, smiles_mecis)]
    ax.legend(handles=meci_handles, title="MECIs", loc="lower left", bbox_to_anchor=(1.02,0.4))

    ax.set_xlabel(f"{reduction_technique} 1")
    ax.set_ylabel(f"{reduction_technique} 2")
    ax.set_title(f"Benzene {reduction_technique} of SeamStress")
    plt.tight_layout()
    plt.show()



def plot_meci_types(reduced_space, idx, mecis, csv_file, n_star, reduction_technique="UMAP",
                    point_size=20, star_size=250, base_colors=None, star_edgecolor="black"):
    """
    Scatter plot colored by MECI type with stars for MECIs.
    """
    if base_colors is None:
        base_colors = ["#1c6f9f","#d85f2b","#3bb12e","#372dca","#e42626","#800080","#00ced1","#15bad4","#60ca97","#1a45bb"]

    df = pd.read_csv(csv_file)
    meci_idx_to_type = dict(zip(df['idx'], df['meci_type']))
    unique_types = sorted(df['meci_type'].unique())
    type_to_color = {t: base_colors[i % len(base_colors)] for i, t in enumerate(unique_types)}

    point_types = [meci_idx_to_type.get(i, "Unknown") for i in idx]
    point_colors = [type_to_color.get(t, "#bdbdbd") for t in point_types]

    # Optional custom star colors
    star_colors_dict = {idx[-4]: "#0075f3", idx[-3]: "#f57812", idx[-2]: "#60e30e", idx[-1]: "#4d21d2"}

    plt.figure(figsize=(7,6))
    plt.scatter(reduced_space[:,0], reduced_space[:,1], c=point_colors, s=point_size, alpha=0.85)

    star_indices = np.arange(len(reduced_space)-n_star, len(reduced_space))
    star_handles = []
    for star_idx in star_indices:
        star_color = star_colors_dict.get(idx[star_idx], point_colors[star_idx])
        h = plt.scatter(
            reduced_space[star_idx,0], reduced_space[star_idx,1],
            marker="*", s=star_size, color=star_color, edgecolor=star_edgecolor,
            linewidth=1.0, zorder=10, label=str(idx[star_idx])
        )
        star_handles.append(h)

    # Legends
    type_patches = [mpatches.Patch(color=color, label=t) for t,color in type_to_color.items()]
    type_legend = plt.legend(handles=type_patches, title="MECI Type", bbox_to_anchor=(1.0,1),
                             loc="upper left", title_fontsize=14, fontsize=12)
    plt.gca().add_artist(type_legend)

    star_legend = plt.legend(handles=star_handles, title="MECIs", bbox_to_anchor=(1.05,0.3),
                             loc="upper left", title_fontsize=14, fontsize=12)

    plt.xlabel(f"{reduction_technique} 1", fontsize=16)
    plt.ylabel(f"{reduction_technique} 2", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Benzene Type 2 Ref {reduction_technique} Spawns colored by MECI Optimization", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
