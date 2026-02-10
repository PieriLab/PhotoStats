import os
import torch
import numpy as np
from ase import Atoms
from ase.io import write
from process_geometries import GeometryDataset

# ----------------------------
# Helper functions
# ----------------------------

def compute_distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)

def compute_inverse_distance_matrix(coords):
    D = compute_distance_matrix(coords)
    with np.errstate(divide='ignore'):
        D_inv = 1.0 / (D + 1e-8)
    np.fill_diagonal(D_inv, 0.0)
    return D_inv

def distance_matrix_average(structures, metric='distance'):
    """
    Compute average distance / inverse distance matrix over structures.
    metric: 'distance' or 'inverse_distance'
    """
    D_total = None
    for atoms in structures:
        coords = atoms.get_positions()
        if metric == 'distance':
            D = compute_distance_matrix(coords)
        elif metric == 'inverse_distance':
            D = compute_inverse_distance_matrix(coords)
        else:
            raise ValueError("metric must be 'distance' or 'inverse_distance'")
        if D_total is None:
            D_total = np.zeros_like(D)
        D_total += D
    return D_total / len(structures)

def reconstruct_coords_from_distance_matrix(D_target, n_iter=5000, lr=1e-2, verbose=True):
    """
    Reconstruct Cartesian coordinates from a target distance matrix using PyTorch.
    Minimize Frobenius norm: ||D(X) - D_target||^2
    """
    n_atoms = D_target.shape[0]
    X = torch.randn((n_atoms, 3), requires_grad=True)
    D_target_t = torch.tensor(D_target, dtype=torch.float32)
    optimizer = torch.optim.Adam([X], lr=lr)

    for step in range(n_iter):
        optimizer.zero_grad()
        diff = X[:, None, :] - X[None, :, :]
        D_current = torch.norm(diff, dim=-1)
        loss = torch.norm(D_current - D_target_t, p='fro') ** 2
        loss.backward()
        optimizer.step()

        if verbose and step % 500 == 0:
            print(f"Step {step}, loss = {loss.item():.6f}")

    return X.detach().numpy()

def compute_cartesian_average(structures):
    n_atoms = len(structures[0])
    avg_pos = np.zeros((n_atoms, 3))
    for atoms in structures:
        avg_pos += atoms.get_positions()
    avg_pos /= len(structures)
    return avg_pos

def compute_medoid(structures):
    """
    Compute medoid: structure with minimal total RMSD to all others.
    Assumes pre-aligned structures.
    """
    n = len(structures)
    total_rmsd = np.zeros(n)
    for i in range(n):
        Ri = structures[i].get_positions()
        for j in range(n):
            Rj = structures[j].get_positions()
            total_rmsd[i] += np.linalg.norm(Ri - Rj)
    medoid_idx = np.argmin(total_rmsd)
    return structures[medoid_idx].get_positions()

# ----------------------------
# Main pipeline
# ----------------------------

def main(xyz_folder, meci_labels_csv, output_folder,
         representation="cartesian", average_method="mean"):
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset
    dataset = GeometryDataset(folder_path=xyz_folder, meci_labels_csv=meci_labels_csv)
    print(f"Loaded {len(dataset)} structures from {xyz_folder}")

    # Group by MECI class
    class_structures = {}
    for atoms, label in zip(dataset.structures, dataset.meci_labels):
        class_structures.setdefault(label, []).append(atoms)

    # Compute averages and save
    for label, structures in class_structures.items():
        print(f"\nProcessing class {label} ({len(structures)} structures)")

        if representation == "cartesian":
            if average_method == "mean":
                avg_coords = compute_cartesian_average(structures)
            elif average_method == "medoid":
                avg_coords = compute_medoid(structures)
            else:
                raise ValueError("Unsupported average_method for cartesian")
        elif representation in ["distance", "inverse_distance"]:
            D_avg = distance_matrix_average(structures, metric=representation)
            if average_method == "mean":
                avg_coords = reconstruct_coords_from_distance_matrix(D_avg)
            elif average_method == "medoid":
                # find structure closest to D_avg in Frobenius norm
                min_idx, min_dist = None, np.inf
                for i, atoms in enumerate(structures):
                    coords = atoms.get_positions()
                    if representation == "distance":
                        D = compute_distance_matrix(coords)
                    else:
                        D = compute_inverse_distance_matrix(coords)
                    dist = np.linalg.norm(D - D_avg, ord='fro')
                    if dist < min_dist:
                        min_idx = i
                        min_dist = dist
                avg_coords = structures[min_idx].get_positions()
            else:
                raise ValueError("Unsupported average_method for distance matrices")
        else:
            raise ValueError("Unsupported representation")

        # Use first structure for atom symbols
        symbols = structures[0].get_chemical_symbols()
        avg_atoms = Atoms(symbols=symbols, positions=avg_coords)

        # Write XYZ
        output_file = os.path.join(output_folder, f"{label}_{representation}_{average_method}.xyz")
        write(output_file, avg_atoms)
        print(f"Saved structure to {output_file}")



if __name__ == "__main__":
    xyz_folder = '/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/aligned_geometries/SeamStress_improper_rotations/benzene'
    meci_labels_csv = '/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/meci_classification/benzene/S1S0/meci_labels_humanlabels.csv'
    output_folder = "./class_averages"

    # options
    representation = "cartesian"  # cartesian, distance, inverse_distance
    average_method = "mean"  # mean, medoid

    main(xyz_folder, meci_labels_csv, output_folder, representation, average_method)
