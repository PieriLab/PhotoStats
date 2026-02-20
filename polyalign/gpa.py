import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from scipy.optimize import linear_sum_assignment
from itertools import permutations, product
from tqdm import tqdm

# ------------------------------
# Utilities
# ------------------------------
def read_xyz(filename):
    atoms, coords = [], []
    with open(filename) as f:
        lines = f.readlines()[2:]  # skip header
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords, dtype=float)

def kabsch(P, Q):
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    P_cent = P - P.mean(0)
    Q_cent = Q - Q.mean(0)
    C = P_cent.T @ Q_cent
    V, S, Wt = np.linalg.svd(C)
    if np.linalg.det(V @ Wt) < 0:
        V[:, -1] *= -1
    R = V @ Wt
    aligned = (P_cent @ R) + Q.mean(0)
    return aligned

def rmsd(P, Q):
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))

def xyz_to_rdkit_mol(atoms, coords, total_charge=0):
    mol = Chem.RWMol()
    z = [Chem.GetPeriodicTable().GetAtomicNumber(a) for a in atoms]
    for Zi in z:
        mol.AddAtom(Chem.Atom(Zi))
    conf = Chem.Conformer(len(coords))
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    # Let RDKit try to guess connectivity
    try:
        Chem.rdDetermineBonds.DetermineConnectivity(mol, charge=total_charge)
        Chem.SanitizeMol(mol)
    except:
        pass
    return mol

# ------------------------------
# Alignment routines
# ------------------------------
def rdkit_substructure_align(ref_mol, test_mol, ref_coords, test_coords):
    matches = test_mol.GetSubstructMatches(ref_mol, uniquify=True)
    if not matches:
        return None, None
    best_rmsd = np.inf
    best_coords = None
    for match in matches:
        P = np.array([test_coords[i] for i in match], dtype=float)
        Q = np.array(ref_coords[:len(match)], dtype=float)
        aligned_sub = kabsch(P, Q)
        # Build full molecule aligned coordinates
        full_aligned = np.array(test_coords, dtype=float)
        for idx, pos in zip(match, aligned_sub):
            full_aligned[idx] = pos
        val = rmsd(full_aligned, ref_coords)
        if val < best_rmsd:
            best_rmsd = val
            best_coords = full_aligned
    return best_coords, best_rmsd

def mcs_hungarian_align(ref_mol, test_mol, ref_coords, test_coords):
    mcs = rdFMCS.FindMCS([ref_mol, test_mol],
                         completeRingsOnly=True,
                         ringMatchesRingOnly=True,
                         matchValences=True)
    if mcs.numAtoms == 0:
        return None, None
    patt = Chem.MolFromSmarts(mcs.smartsString)
    ref_match = ref_mol.GetSubstructMatch(patt)
    test_match = test_mol.GetSubstructMatch(patt)
    if not ref_match or not test_match:
        return None, None
    # MCS aligned first
    P = np.array([test_coords[i] for i in test_match], dtype=float)
    Q = np.array([ref_coords[i] for i in ref_match], dtype=float)
    aligned = kabsch(P, Q)
    # Assign remaining atoms using Hungarian on distance
    remaining_test = [i for i in range(len(test_coords)) if i not in test_match]
    remaining_ref = [i for i in range(len(ref_coords)) if i not in ref_match]
    if remaining_test and remaining_ref:
        D = np.linalg.norm(
            np.array([aligned[i] for i in test_match])[...,None,:] - np.array([ref_coords[i] for i in ref_match])[None,...,:], axis=2
        )
        row, col = linear_sum_assignment(D)
        # Map remaining based on closest distances
        for i_test, i_ref in zip(remaining_test, remaining_ref):
            aligned_full = np.array(test_coords, dtype=float)
            aligned_full[i_test] = test_coords[i_test]  # will refine with full Kabsch
        aligned_full = kabsch(aligned_full, ref_coords)
    else:
        aligned_full = np.array(test_coords, dtype=float)
        aligned_full = kabsch(aligned_full, ref_coords)
    return aligned_full, rmsd(aligned_full, ref_coords)

def inertia_hungarian_align(ref_coords, test_coords):
    ref_coords = np.array(ref_coords, dtype=float)
    test_coords = np.array(test_coords, dtype=float)

    def inertia_axes(X):
        Xc = X - X.mean(0)
        I = Xc.T @ Xc
        _, eigvecs = np.linalg.eigh(I)
        return eigvecs

    ref_axes = inertia_axes(ref_coords)
    test_axes = inertia_axes(test_coords)
    ref_rot = ref_coords @ ref_axes
    test_rot = test_coords @ test_axes
    D = np.linalg.norm(ref_rot[:, None, :] - test_rot[None, :, :], axis=2)
    row, col = linear_sum_assignment(D)
    P = test_coords[col]
    aligned_full = kabsch(P, ref_coords)
    return aligned_full, rmsd(aligned_full, ref_coords)

# ------------------------------
# One-step alignment
# ------------------------------
def align_one(ref_atoms, ref_coords, test_atoms, test_coords):
    ref_mol = xyz_to_rdkit_mol(ref_atoms, ref_coords)
    test_mol = xyz_to_rdkit_mol(test_atoms, test_coords)
    # 1️⃣ Try substructure
    aligned, val = rdkit_substructure_align(ref_mol, test_mol, ref_coords, test_coords)
    if aligned is not None:
        return aligned, val
    # 2️⃣ MCS + Hungarian
    aligned_mcs, val_mcs = mcs_hungarian_align(ref_mol, test_mol, ref_coords, test_coords)
    # 3️⃣ Inertia + Hungarian
    aligned_inertia, val_inertia = inertia_hungarian_align(ref_coords, test_coords)
    # Choose best RMSD
    if val_mcs is None or val_inertia < val_mcs:
        return aligned_inertia, val_inertia
    else:
        return aligned_mcs, val_mcs

# ------------------------------
# GPA function
# ------------------------------
def gpa(input_folder, output_folder, max_iter=5, tol=1e-4, save_xyz=True):
    """
    Perform iterative GPA alignment on all XYZ structures in a folder.
    Iteratively aligns all structures to the mean structure until convergence
    or max_iter is reached. Prints RMSD to mean at each iteration.

    Returns:
        aligned_coords: ndarray of shape (n_structures, n_atoms, 3)
        files: list of filenames corresponding to aligned_coords
    """
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(".xyz")]
    n_files = len(files)
    aligned_coords = []

    # Read all molecules first
    all_atoms = []
    all_coords = []
    for f in tqdm(files, desc="Reading XYZs"):
        atoms, coords = read_xyz(os.path.join(input_folder, f))
        all_atoms.append(atoms)
        all_coords.append(coords)
    n_atoms = len(all_coords[0])

    # Initialize aligned_coords with original coordinates
    aligned_coords = [np.array(c, dtype=float) for c in all_coords]

    for iteration in range(max_iter):
        print(f"\nIteration {iteration+1}")

        # 1️⃣ Compute mean structure
        mean_structure = np.mean(np.stack(aligned_coords), axis=0)

        # 2️⃣ Align each structure to mean
        rmsd_to_mean = []
        for i in range(n_files):
            try:
                aligned, rmsd_val = align_one(all_atoms[0], mean_structure,
                                              all_atoms[i], aligned_coords[i])
                if aligned is None:
                    aligned = aligned_coords[i]
            except Exception as e:
                print(f"Error aligning {files[i]}: {e}")
                aligned = aligned_coords[i]

            aligned_coords[i] = np.array(aligned, dtype=float)
            rmsd_to_mean.append(rmsd(aligned_coords[i], mean_structure))

        # 3️⃣ Print convergence info
        avg_rmsd = np.mean(rmsd_to_mean)
        max_rmsd = np.max(rmsd_to_mean)
        print(f"Avg RMSD to mean: {avg_rmsd:.6f}, Max RMSD: {max_rmsd:.6f}")

        # 4️⃣ Check convergence
        if avg_rmsd < tol:
            print("Converged!")
            break

    # 5️⃣ Save aligned XYZs
    if save_xyz:
        for i, f in enumerate(files):
            out_path = os.path.join(output_folder, f"aligned_{f}")
            with open(out_path, "w") as out_f:
                out_f.write(f"{n_atoms}\nAligned GPA iteration {iteration+1}\n")
                for atom, pos in zip(all_atoms[i], aligned_coords[i]):
                    out_f.write(f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    aligned_coords = np.array(aligned_coords, dtype=float)
    return aligned_coords, files


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    INPUT_FOLDER = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/polyalign/unaligned"
    OUTPUT_FOLDER = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/polyalign/aligned_output"
    aligned_coords, files = gpa(INPUT_FOLDER, OUTPUT_FOLDER)
    print("Aligned coordinates shape:", aligned_coords.shape)
