import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_molecule_2d_3d_bonds(df, vector_type="dev", scale=1.0, title=None, bond_cutoff=1.6):
    """
    2D molecular plot (XY-plane) colored by vector magnitude.
    Bonds are drawn if 3D distance < bond_cutoff.
    """
    # 3D coordinates
    X = df["meci_x"].to_numpy()
    Y = df["meci_y"].to_numpy()
    Z = df["meci_z"].to_numpy()

    # Compute vector components
    if vector_type == "dev":
        U = df["dev_x"].to_numpy()
        V = df["dev_y"].to_numpy()
    elif vector_type == "variance":
        U = np.sqrt(df["var_x"].to_numpy())
        V = np.sqrt(df["var_y"].to_numpy())
    elif vector_type == "skew":
        U = df["skew_x"].to_numpy() * np.sqrt(df["var_x"].to_numpy())
        V = df["skew_y"].to_numpy() * np.sqrt(df["var_y"].to_numpy())
    else:
        raise ValueError(f"Unknown vector_type: {vector_type}")

    # Vector magnitudes for coloring
    vec_magnitudes = np.linalg.norm(np.stack([U, V], axis=1), axis=1)
    norm = plt.Normalize(vec_magnitudes.min(), vec_magnitudes.max())
    cmap = plt.cm.viridis
    colors = cmap(norm(vec_magnitudes))

    # Atom sizes
    element_sizes = {"C": 120, "H": 50}
    sizes = [element_sizes.get(e, 80) for e in df["element"]]

    # Plotting
    n_atoms = len(X)
    fig, ax = plt.subplots(figsize=(8,8))

    # Draw bonds using 3D distance
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist_3d = np.linalg.norm([X[i]-X[j], Y[i]-Y[j], Z[i]-Z[j]])
            if dist_3d <= bond_cutoff:
                ax.plot([X[i], X[j]], [Y[i], Y[j]], color='gray', lw=2, zorder=1)

    ax.scatter(X, Y, s=sizes, color=colors, edgecolors='k', zorder=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"{vector_type.capitalize()} magnitude", rotation=270, labelpad=15)

    ax.set_aspect('equal')
    ax.axis('off')
    if title is None:
        title = f"{vector_type.capitalize()} (2D)"
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    csv_file = "data/class_stat_analysis/benzene_class_averages/Type 3_atomwise_stats.csv"
    df = pd.read_csv(csv_file)

    plot_molecule_2d_3d_bonds(df, vector_type="dev", scale=1.0, title="Mean Deviation from MECI Angstrom")

    plot_molecule_2d_3d_bonds(df, vector_type="variance", scale=1.0, title="Standard Deviation Angstrom")

    plot_molecule_2d_3d_bonds(df, vector_type="skew", scale=1.0, title="Skewness")
