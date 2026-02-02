import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

def plot_spawns_vs_mecis(Molecule_name=None, X_spawns=None, X_mecis=None,
                         smiles_spawns=None, smiles_mecis=None, idx_mecis=None,
                         reduction_technique="UMAP", style_file=None, save_path=None):
    """
    Scatter plot of spawns vs MECIs, colored by SMILES.
    If save_path is provided, the figure is saved instead of shown.
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

    # Plot spawns
    for smi in unique_smiles:
        mask = [s == smi for s in smiles_spawns]
        ax.scatter(
            X_spawns[mask,0],
            X_spawns[mask,1],
            color=smiles_to_color[smi],
            s=20,
            alpha=0.6
        )

    # Plot MECIs
    for (x,y), smi in zip(X_mecis, smiles_mecis):
        ax.scatter(x, y, marker="*", s=200, color=smiles_to_color[smi], edgecolor="k", linewidth=0.8)

    # Legends
    smiles_handles = [mlines.Line2D([], [], marker='o', linestyle='', color=smiles_to_color[smi], label=smi, markersize=8) for smi in unique_smiles]
    legend_smiles = ax.legend(handles=smiles_handles, title="SMILES", loc="upper left", bbox_to_anchor=(0.97,1), fontsize=8)
    ax.add_artist(legend_smiles)

    meci_handles = [mlines.Line2D([], [], marker='*', linestyle='', color=smiles_to_color[smi], markeredgecolor='k', markeredgewidth=0.8, label=name, markersize=12) for name, smi in zip(idx_mecis, smiles_mecis)]
    ax.legend(handles=meci_handles, title="MECIs", loc="lower left", bbox_to_anchor=(1.02,0.4))

    ax.set_xlabel(f"{reduction_technique} 1")
    ax.set_ylabel(f"{reduction_technique} 2")
    if Molecule_name:
        ax.set_title(f"{Molecule_name} {reduction_technique} of SeamStress")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved SMILES plot to {save_path}")
    else:
        plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

def plot_meci_types(Molecule_name=None, reduced_space=None, idx=None, mecis=None, csv_file=None,
                    reduction_technique="UMAP", point_size=20, star_size=250, base_colors=None,
                    star_edgecolor="black", save_path=None):
    """
    Scatter plot colored by MECI type with stars for MECIs.
    If save_path is provided, the figure is saved instead of shown.
    
    Parameters
    ----------
    Molecule_name : str, optional
        Name of the molecule for plot title.
    reduced_space : np.ndarray
        2D array of reduced coordinates (n_points x 2).
    idx : list
        List of identifiers for each point in reduced_space.
    mecis : list
        List of MECI file paths used to identify stars.
    csv_file : str or Path
        CSV file with columns ['idx', 'meci_type'] for coloring.
    reduction_technique : str
        Dimensionality reduction technique used for labeling axes.
    point_size : int
        Size of regular points (spawns).
    star_size : int
        Size of MECI points (stars).
    base_colors : list of str
        List of colors to cycle through for MECI types.
    star_edgecolor : str
        Edge color for stars.
    save_path : str or Path, optional
        Path to save the figure. If None, plt.show() is called.
    """
    if base_colors is None:
        base_colors = ["#1c6f9f","#d85f2b","#3bb12e","#372dca","#e42626",
                       "#800080","#00ced1","#15bad4","#60ca97","#1a45bb"]

    # Load MECI types from CSV
    df = pd.read_csv(csv_file)
    n_star = len(mecis)
    meci_idx_to_type = dict(zip(df['idx'], df['meci_type']))
    unique_types = sorted(df['meci_type'].unique())
    type_to_color = {t: base_colors[i % len(base_colors)] for i, t in enumerate(unique_types)}

    # Assign colors to all points
    point_types = [meci_idx_to_type.get(i, "Unknown") for i in idx]
    point_colors = [type_to_color.get(t, "#bdbdbd") for t in point_types]

    # Optional custom star colors for the last 4 MECIs
    star_colors_dict = {idx[-4]: "#0075f3", idx[-3]: "#f57812", idx[-2]: "#60e30e", idx[-1]: "#4d21d2"}

    # Start figure
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(reduced_space[:,0], reduced_space[:,1], c=point_colors, s=point_size, alpha=0.85)

    # Plot stars
    star_indices = np.arange(len(reduced_space)-n_star, len(reduced_space))
    star_handles = []
    for star_idx in star_indices:
        star_color = star_colors_dict.get(idx[star_idx], point_colors[star_idx])
        h = ax.scatter(
            reduced_space[star_idx,0], reduced_space[star_idx,1],
            marker="*", s=star_size, color=star_color, edgecolor=star_edgecolor,
            linewidth=1.0, zorder=10, label=str(idx[star_idx])
        )
        star_handles.append(h)

    # Legends
    type_patches = [mpatches.Patch(color=color, label=t) for t,color in type_to_color.items()]
    type_legend = ax.legend(handles=type_patches, title="MECI Type", bbox_to_anchor=(1.0,1),
                             loc="upper left", title_fontsize=14, fontsize=12)
    ax.add_artist(type_legend)

    star_legend = ax.legend(handles=star_handles, title="MECIs", bbox_to_anchor=(1.05,0.3),
                             loc="upper left", title_fontsize=14, fontsize=12)

    # Labels, title, ticks
    ax.set_xlabel(f"{reduction_technique} 1", fontsize=16)
    ax.set_ylabel(f"{reduction_technique} 2", fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if Molecule_name:
        ax.set_title(f"{Molecule_name} Type 2 Ref {reduction_technique} Spawns colored by MECI Optimization", fontsize=16)

    ax.grid(False)
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved MECI types plot to {save_path}")
    else:
        plt.show()
