import numpy as np
from itertools import permutations, product
from collections import defaultdict
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# ------------------------------
# User-defined variables
# ------------------------------
REFERENCE_FOLDER = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/polyalign/gpa_aligned_output"
TEST_FOLDER = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/polyalign/unaligned"
OUTPUT_FOLDER = "closest_aligned_output"
NUM_PROCESSES = cpu_count()

# ------------------------------
# Functions
# ------------------------------
def read_xyz(filename):
    atoms, coords = [], []
    with open(filename) as f:
        lines = f.readlines()[2:]
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(atoms), np.array(coords)

def write_xyz(filename, atoms, coords, comment=""):
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")
        for atom, c in zip(atoms, coords):
            f.write(f"{atom} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

def kabsch(P, Q):
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)
    C = np.dot(P_cent.T, Q_cent)
    V, S, Wt = np.linalg.svd(C)
    #if np.linalg.det(V @ Wt) < 0:
    #    V[:, -1] *= -1
    U = np.dot(V, Wt)
    return P_cent @ U + Q.mean(axis=0)

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

# ------------------------------
# Align a single test structure to all references
# ------------------------------
def align_to_references(args):
    test_file, ref_files, output_folder = args
    test_atoms, test_coords = read_xyz(test_file)

    perms = atom_type_permutations(test_atoms)
    min_rmsd = float('inf')
    best_coords = None
    best_ref = None

    for ref_file in ref_files:
        ref_atoms, ref_coords = read_xyz(ref_file)
        if list(test_atoms) != list(ref_atoms):
            continue

        for perm in perms:
            permuted = test_coords[perm]
            aligned = kabsch(permuted, ref_coords)
            r = rmsd(aligned, ref_coords)
            if r < min_rmsd:
                min_rmsd = r
                best_coords = aligned
                best_ref = os.path.basename(ref_file)

    if best_coords is not None:
        output_file = os.path.join(output_folder, f"aligned_{os.path.basename(test_file)}")
        write_xyz(output_file, test_atoms, best_coords, comment=f"Aligned to {best_ref}, RMSD {min_rmsd:.5f}")
        return os.path.basename(test_file), best_ref, min_rmsd

    return None

# ------------------------------
# Main execution
# ------------------------------
def align_folder_to_references(reference_folder, test_folder, output_folder="aligned_output"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ref_files = sorted([os.path.join(reference_folder, f) for f in os.listdir(reference_folder) if f.endswith(".xyz")])
    test_files = sorted([os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(".xyz")])

    args_list = [(tf, ref_files, output_folder) for tf in test_files]

    results = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for r in tqdm(pool.imap(align_to_references, args_list), total=len(args_list), desc="Aligning tests"):
            if r is not None:
                results.append(r)

    return results

# ------------------------------
if __name__ == "__main__":
    mp.set_start_method("forkserver")
    results = align_folder_to_references(REFERENCE_FOLDER, TEST_FOLDER, OUTPUT_FOLDER)
    for test_name, ref_name, rmsd_val in results:
        print(f"{test_name} best aligned to {ref_name}, RMSD {rmsd_val:.5f}")
