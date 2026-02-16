import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis


def prepare_vectors(df, vector_type="dev"):
    X = df["meci_x"].to_numpy()
    Y = df["meci_y"].to_numpy()
    Z = df["meci_z"].to_numpy()

    if vector_type == "dev":
        U = df["dev_x"].to_numpy()
        V = df["dev_y"].to_numpy()
        W = df["dev_z"].to_numpy()
    elif vector_type == "variance":
        U = np.sqrt(df["var_x"].to_numpy())
        V = np.sqrt(df["var_y"].to_numpy())
        W = np.sqrt(df["var_z"].to_numpy())
    elif vector_type == "skew":
        U = df["skew_x"].to_numpy() * np.sqrt(df["var_x"].to_numpy())
        V = df["skew_y"].to_numpy() * np.sqrt(df["var_y"].to_numpy())
        W = df["skew_z"].to_numpy() * np.sqrt(df["var_z"].to_numpy())
    else:
        raise ValueError(f"Unknown vector_type: {vector_type}")

    return X, Y, Z, U, V, W   # <-- MUST be 6 separate arrays

def draw_bonds(ax, X, Y, Z, cutoff=1.6):
    """
    Draw bonds between atoms if distance < cutoff (in Å).
    Simple approximate connectivity.
    """
    n_atoms = len(X)
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm([X[i]-X[j], Y[i]-Y[j], Z[i]-Z[j]])
            if dist <= cutoff:
                ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]], color='gray', linewidth=2)

def plot_molecule_vectors(df, vector_type="dev", scale=1.0, title=None, bond_cutoff=1.6):
    X, Y, Z, U, V, W = prepare_vectors(df, vector_type)

    if title is None:
        title = vector_type.capitalize()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Map elements to color and size
    element_colors = {"C": "black", "H": "gray"}  # add more if needed
    element_sizes = {"C": 120, "H": 50}

    # Plot atoms with colors/sizes from element
    for i, elem in enumerate(df["element"]):
        ax.scatter(X[i], Y[i], Z[i],
                   color=element_colors.get(elem, "blue"),
                   s=element_sizes.get(elem, 80))

    # Plot bonds
    draw_bonds(ax, X, Y, Z, cutoff=bond_cutoff)

    # Vector magnitudes for coloring
    vec_magnitudes = np.linalg.norm(np.stack([U, V, W], axis=1), axis=1)
    norm = plt.Normalize(vec_magnitudes.min(), vec_magnitudes.max())
    cmap = plt.cm.viridis  # or any other colormap
    vector_colors = cmap(norm(vec_magnitudes))

    # Plot vectors with magnitude-dependent color
    for i in range(len(X)):
        ax.quiver(X[i], Y[i], Z[i],
                  U[i]*scale, V[i]*scale, W[i]*scale,
                  color=vector_colors[i], length=1.0, normalize=False)

    # Remove grid, axes ticks, and set aspect
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_facecolor('white')
    ax.set_box_aspect([1,1,.4])  # equal aspect ratio

    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    csv_file = "data/class_stat_analysis/benzene_class_averages/Type 3_atomwise_stats.csv"  # replace with your path
    df = pd.read_csv(csv_file)

    plot_molecule_vectors(df, vector_type="dev", scale=1.0, title="Deviation from MECI", bond_cutoff=1.6)

    plot_molecule_vectors(df, vector_type="variance", scale=1.0, title="Variance", bond_cutoff=1.6)

    plot_molecule_vectors(df, vector_type="skew", scale=1.0, title="Skewness", bond_cutoff=1.6)
