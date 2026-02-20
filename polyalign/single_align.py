import numpy as np
from itertools import permutations, product
from collections import defaultdict
import os
import umap
import plotly.express as px
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import multiprocessing as mp 

# ------------------------------
# User-defined variables
# ------------------------------
REFERENCE_FILE = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/polyalign/references/Type_3.xyz"
FOLDER_PATH = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/polyalign/unaligned"
OUTPUT_FOLDER = "aligned_output"
NUM_PROCESSES = cpu_count()  # Use all available cores

# ------------------------------
# Functions
# ------------------------------
def read_xyz(filename):
    atoms = []
    coords = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            atom, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(atom)
            coords.append([x, y, z])
    return np.array(atoms), np.array(coords)

def kabsch(P, Q):
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)
    C = np.dot(P_cent.T, Q_cent)
    V, S, Wt = np.linalg.svd(C)
    #if (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0:
    #    V[:, -1] *= -1
    U = np.dot(V, Wt)
    P_rot = np.dot(P_cent, U)
    return P_rot + Q.mean(axis=0)

def rmsd(P, Q):
    diff = P - Q
    return np.sqrt((diff * diff).sum() / len(P))

def atom_type_permutations(atoms):
    type_indices = defaultdict(list)
    for i, atom in enumerate(atoms):
        type_indices[atom].append(i)
    type_perms = {t: list(permutations(idxs)) for t, idxs in type_indices.items()}
    all_combos = product(*type_perms.values())
    perms_list = []
    for combo in all_combos:
        perm_indices = [0] * len(atoms)
        for indices, perm in zip(type_indices.values(), combo):
            for orig_idx, perm_idx in zip(indices, perm):
                perm_indices[orig_idx] = perm_idx
        perms_list.append(perm_indices)
    return perms_list

def align_single_file(args):
    """Align a single XYZ file to reference (for multiprocessing)."""
    xyz_file, ref_atoms, ref_coords, folder_path, output_folder = args
    mobile_file = os.path.join(folder_path, xyz_file)
    mob_atoms, mob_coords = read_xyz(mobile_file)
    if list(ref_atoms) != list(mob_atoms):
        return None  # Skip if atom types don't match

    perms = atom_type_permutations(mob_atoms)
    min_rmsd = float('inf')
    best_perm_coords = None

    for perm in perms:
        permuted_coords = mob_coords[perm]
        aligned_coords = kabsch(permuted_coords, ref_coords)
        rmsd_val = rmsd(aligned_coords, ref_coords)
        if rmsd_val < min_rmsd:
            min_rmsd = rmsd_val
            best_perm_coords = aligned_coords

    # Save aligned XYZ
    output_file = os.path.join(output_folder, f"aligned_{xyz_file}")
    with open(output_file, 'w') as f:
        f.write(f"{len(best_perm_coords)}\n")
        f.write(f"Aligned to reference, RMSD {min_rmsd:.5f}\n")
        for atom, coord in zip(ref_atoms, best_perm_coords):
            f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

    return xyz_file, best_perm_coords.flatten(), min_rmsd

def align_folder(reference_file, folder_path, output_folder="aligned_output"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ref_atoms, ref_coords = read_xyz(reference_file)
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".xyz")]

    # Prepare arguments for pool
    args_list = [(f, ref_atoms, ref_coords, folder_path, output_folder) for f in all_files]

    vectors = []
    filenames = []
    rmsd_list = []

    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap(align_single_file, args_list), total=len(args_list), desc="Aligning XYZs"):
            if result is None:
                continue
            fname, vec, rmsd_val = result
            filenames.append(fname)
            vectors.append(vec)
            rmsd_list.append(rmsd_val)

    return np.array(vectors), filenames, rmsd_list

def plot_umap(vectors, labels):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(vectors)
    fig = px.scatter(
        x=embedding[:,0],
        y=embedding[:,1],
        text=labels,
        color=labels,
        title="2D UMAP of Aligned Geometries"
    )
    fig.show()

# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    mp.set_start_method("forkserver")  # Use "spawn" on Mac by default, "forkserver" is safer
    vectors, files, rmsd_list = align_folder(REFERENCE_FILE, FOLDER_PATH, OUTPUT_FOLDER)
    plot_umap(vectors, files)
