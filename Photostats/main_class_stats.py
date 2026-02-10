import os
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from process_geometries import GeometryDataset
from scipy.stats import skew, kurtosis

# ----------------------------
# Helper functions
# ----------------------------

def compute_cartesian_average(structures):
    """Compute per-atom mean positions."""
    n_atoms = len(structures[0])
    avg_pos = np.zeros((n_atoms, 3))
    for atoms in structures:
        avg_pos += atoms.get_positions()
    avg_pos /= len(structures)
    return avg_pos

def compute_medoid(structures):
    """Compute medoid: structure with minimal total RMSD to all others."""
    n = len(structures)
    total_rmsd = np.zeros(n)
    for i in range(n):
        Ri = structures[i].get_positions()
        for j in range(n):
            Rj = structures[j].get_positions()
            total_rmsd[i] += np.linalg.norm(Ri - Rj)
    medoid_idx = np.argmin(total_rmsd)
    return structures[medoid_idx].get_positions()

from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd

def compute_atomwise_stats_vs_meci(structures, ref_coords):
    """
    Compute per-atom statistics relative to a reference (MECI) geometry.

    Returns a pandas DataFrame with:
      - Atom type
      - MECI coordinates
      - Average coordinates of the class
      - Deviation from MECI to average
      - Component-wise variance, skewness, kurtosis relative to MECI
      - Norms for deviation, variance, skewness
    """
    n_atoms = len(structures[0])
    all_coords = np.array([atoms.get_positions() for atoms in structures])  # (n_structures, n_atoms, 3)

    # Atom symbols
    symbols = structures[0].get_chemical_symbols()  # assume same order for all structures

    # Compute per-atom average coordinates
    avg_coords = np.mean(all_coords, axis=0)  # (n_atoms,3)

    # Deviation from MECI to average
    dev_from_meci = avg_coords - ref_coords  # (n_atoms,3)
    dev_norm = np.linalg.norm(dev_from_meci, axis=1)

    # Displacement from MECI for all structures
    displacement = all_coords - ref_coords[None, :, :]  # (n_structures, n_atoms, 3)
    displacement_mags = np.linalg.norm(displacement, axis=-1)  # (n_structures, n_atoms)

    # Component-wise variance
    var_x = np.var(displacement[:, :, 0], axis=0)
    var_y = np.var(displacement[:, :, 1], axis=0)
    var_z = np.var(displacement[:, :, 2], axis=0)
    var_norm = np.linalg.norm(np.stack([var_x, var_y, var_z], axis=1), axis=1)

    # Component-wise skewness
    skew_x = skew(displacement[:, :, 0], axis=0)
    skew_y = skew(displacement[:, :, 1], axis=0)
    skew_z = skew(displacement[:, :, 2], axis=0)
    skew_norm = np.linalg.norm(np.stack([skew_x, skew_y, skew_z], axis=1), axis=1)

    # Kurtosis (magnitude-based)
    kurts = kurtosis(displacement_mags, axis=0)  # excess kurtosis

    # Assemble DataFrame
    df = pd.DataFrame({
        "atom_index": np.arange(n_atoms),
        "element": symbols,
        "meci_x": ref_coords[:, 0],
        "meci_y": ref_coords[:, 1],
        "meci_z": ref_coords[:, 2],
        "avg_x": avg_coords[:, 0],
        "avg_y": avg_coords[:, 1],
        "avg_z": avg_coords[:, 2],
        "dev_x": dev_from_meci[:, 0],
        "dev_y": dev_from_meci[:, 1],
        "dev_z": dev_from_meci[:, 2],
        "dev_norm": dev_norm,
        "var_x": var_x,
        "var_y": var_y,
        "var_z": var_z,
        "var_norm": var_norm,
        "skew_x": skew_x,
        "skew_y": skew_y,
        "skew_z": skew_z,
        "skew_norm": skew_norm,
        "kurtosis": kurts
    })

    return df





# ----------------------------
# Main pipeline
# ----------------------------

def main(xyz_folder, meci_labels_csv, meci_class_meci_dict, output_folder,
         average_method="mean"):
    """
    meci_class_meci_dict: dictionary mapping class label -> path to reference MECI XYZ
    e.g. {"Type 1": "MECI_Type1.xyz", "Type 2": "MECI_Type2.xyz"}
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset
    dataset = GeometryDataset(folder_path=xyz_folder, meci_labels_csv=meci_labels_csv)
    print(f"Loaded {len(dataset)} structures from {xyz_folder}")

    # Get unique classes from CSV
    unique_classes = set(dataset.meci_labels)
    print("Classes in dataset:", unique_classes)

    # Process each class
    for class_name in unique_classes:
        # Filter structures for this class
        structures = [atoms for atoms, label in zip(dataset.structures, dataset.meci_labels)
                      if label == class_name]
        if len(structures) == 0:
            print(f"Skipping empty class {class_name}")
            continue
        print(f"\nProcessing class {class_name} ({len(structures)} structures)")

        # Compute average geometry
        if average_method == "mean":
            avg_coords = compute_cartesian_average(structures)
        elif average_method == "medoid":
            avg_coords = compute_medoid(structures)
        else:
            raise ValueError("Unsupported average_method")

        # Load reference MECI coordinates for this class
        ref_path = meci_class_meci_dict.get(class_name, None)
        if ref_path is not None and os.path.exists(ref_path):
            ref_atoms = read(ref_path)
            ref_coords = ref_atoms.get_positions()
        else:
            print(f"Reference MECI for class {class_name} not found. Using first structure as reference.")
            ref_coords = structures[0].get_positions()

        # Compute per-atom stats
        df_stats = compute_atomwise_stats_vs_meci(structures, ref_coords=ref_coords)

        # Save CSV
        csv_file = os.path.join(output_folder, f"{class_name}_atomwise_stats.csv")
        df_stats.to_csv(csv_file, index=False)
        print(f"Saved per-atom stats to {csv_file}")

        # Save average geometry as XYZ
        symbols = structures[0].get_chemical_symbols()
        avg_atoms = Atoms(symbols=symbols, positions=avg_coords)
        xyz_file = os.path.join(output_folder, f"{class_name}_average.xyz")
        write(xyz_file, avg_atoms)
        print(f"Saved average geometry to {xyz_file}")


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    xyz_folder = '/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/aligned_geometries/SeamStress_improper_rotations/benzene'
    meci_labels_csv = '/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/meci_classification/benzene/S1S0/meci_labels_humanlabels.csv'
    output_folder = "./class_averages"

    # Map each CSV class to a MECI XYZ file
    meci_class_meci_dict = {
        "Type 1": "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/raw_geometries/meci/benzene_main_meci/Type 1.xyz",
        "Type 2": "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/raw_geometries/meci/benzene_main_meci/Type 2.xyz",
        "Type 3": "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/raw_geometries/meci/benzene_main_meci/Type 3.xyz"
    }

    average_method = "mean"  # or "medoid"

    main(xyz_folder, meci_labels_csv, meci_class_meci_dict, output_folder, average_method)
