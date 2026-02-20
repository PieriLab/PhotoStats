import numpy as np
from itertools import permutations, product
from collections import defaultdict
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDetermineBonds

import numpy as np 
from rdkit.Chem import rdFMCS

from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import Image, display


# ------------------------------
# User-defined variables
# ------------------------------
REFERENCE_FOLDER = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/polyalign/ethylene/gpa_ref"
TEST_FOLDER = "/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/polyalign/ethylene/unaligned"
OUTPUT_FOLDER = "closest_aligned_MCS_output"
NUM_PROCESSES = cpu_count()

# ------------------------------
# Functions
# ------------------------------

def xyz_to_rdkit_mol(xyz_file, total_charge):
    """
    Convert an XYZ file with explicit hydrogens to an RDKit Mol,
    preserving 3D coordinates and determining connectivity automatically.
    

    Parameters:
        xyz_file: path to XYZ file
        total_charge: total molecular charge (default 0)
    
    Returns:
        RDKit Mol object
    """
    # --- Read XYZ ---
    with open(xyz_file) as f:
        lines = f.readlines()[2:]  # skip atom count + comment
    atoms, coords = [], []
    for line in lines:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    coords = np.array(coords)
    
    # --- Create empty molecule with atoms ---
    mol = Chem.RWMol()
    z = [Chem.GetPeriodicTable().GetAtomicNumber(a) for a in atoms]
    for Zi in z:
        mol.AddAtom(Chem.Atom(Zi))
    
    # --- Add coordinates ---
    conf = Chem.Conformer(len(coords))
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    
    # --- Determine connectivity using RDKit's bond perception ---
    Chem.rdDetermineBonds.DetermineConnectivity(mol, charge=total_charge)
    
    # --- Sanitize molecule ---
    Chem.SanitizeMol(mol)
    
    return mol

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
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

# ------------------------------
# Align a single test structure to all references using MCS
# ------------------------------
from scipy.optimize import linear_sum_assignment

from scipy.optimize import linear_sum_assignment

def align_to_references_mcs_hungarian_final(args):
    """
    Align a single test XYZ to reference XYZs using:
    1) MCS-based Kabsch
    2) Propagate alignment to all atoms
    3) Hungarian algorithm on all atoms of the same type
    4) Final Kabsch using Hungarian mapping
    5) RMSD computation
    6) Optional Hungarian reordering for closest reference
    Designed for multiprocessing with args = (test_file, ref_files, output_folder)
    """
    test_file, ref_files, output_folder = args

    test_atoms, test_coords = read_xyz(test_file)
    test_mol = xyz_to_rdkit_mol(test_file, total_charge=0)
    
    min_rmsd = float('inf')
    best_coords = None
    best_ref = None
    best_full_map_test = None
    best_full_map_ref = None
    best_ref_coords = None

    # -----------------------------
    # Loop over reference molecules
    # -----------------------------
    for ref_file in ref_files:
        ref_atoms, ref_coords = read_xyz(ref_file)
        if len(test_atoms) != len(ref_atoms):
            continue
        
        ref_mol = xyz_to_rdkit_mol(ref_file, total_charge=0)
        
        # --- Step 1: Find MCS ---
        mcs_result = rdFMCS.FindMCS([test_mol, ref_mol])
        mcs_smarts = mcs_result.smartsString
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)
        
        matches_test = test_mol.GetSubstructMatches(mcs_mol, uniquify=False)
        matches_ref = ref_mol.GetSubstructMatches(mcs_mol, uniquify=False)
        
        for m_test in matches_test:
            for m_ref in matches_ref:
                # --- Step 2: Initial Kabsch alignment on MCS ---
                P = test_coords[list(m_test)]
                Q = ref_coords[list(m_ref)]
                P_aligned = kabsch(P, Q)
                
                # --- Step 3: Propagate rotation/translation to all atoms ---
                P_cent = P - P.mean(axis=0)
                Q_cent = Q - Q.mean(axis=0)
                C = np.dot(P_cent.T, Q_cent)
                V, S, Wt = np.linalg.svd(C)
                U = np.dot(V, Wt)
                all_aligned = (test_coords - test_coords.mean(axis=0)) @ U + Q.mean(axis=0)
                
                # --- Step 4: Hungarian matching for all atoms of same type ---
                unique_types = set(test_atoms)
                full_map_test = []
                full_map_ref = []
                
                for atom_type in unique_types:
                    test_idx = [i for i, a in enumerate(test_atoms) if a == atom_type]
                    ref_idx = [i for i, a in enumerate(ref_atoms) if a == atom_type]

                    cost_matrix = np.zeros((len(test_idx), len(ref_idx)))
                    for i, ti in enumerate(test_idx):
                        for j, rj in enumerate(ref_idx):
                            cost_matrix[i, j] = np.linalg.norm(all_aligned[ti] - ref_coords[rj])

                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    full_map_test.extend([test_idx[i] for i in row_ind])
                    full_map_ref.extend([ref_idx[j] for j in col_ind])

                # --- Step 5: Final Kabsch alignment on fully mapped atoms ---
                P_full = all_aligned[full_map_test]
                Q_full = ref_coords[full_map_ref]
                P_cent = P_full - P_full.mean(axis=0)
                Q_cent = Q_full - Q_full.mean(axis=0)
                C = np.dot(P_cent.T, Q_cent)
                V, S, Wt = np.linalg.svd(C)
                U = np.dot(V, Wt)
                full_aligned_final = (all_aligned - all_aligned.mean(axis=0)) @ U + Q_full.mean(axis=0)

                # --- Step 6: Compute RMSD ---
                rmsd_val = rmsd(full_aligned_final[full_map_test], ref_coords[full_map_ref])

                # Save if best
                if rmsd_val < min_rmsd:
                    min_rmsd = rmsd_val
                    best_coords = full_aligned_final
                    best_ref = os.path.basename(ref_file)
                    best_full_map_test = full_map_test.copy()
                    best_full_map_ref = full_map_ref.copy()
                    best_ref_coords = ref_coords.copy()

    # -----------------------------
    # Step 7: Optional Hungarian reordering to reference for output
    # -----------------------------
    if best_coords is not None:
        aligned_hungarian_output = np.zeros_like(best_coords)
        for test_idx, ref_idx in zip(best_full_map_test, best_full_map_ref):
            aligned_hungarian_output[ref_idx] = best_coords[test_idx]
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = os.path.join(output_folder, f"aligned_{os.path.basename(test_file)}")
        write_xyz(output_file, test_atoms, aligned_hungarian_output, comment=f"Aligned to {best_ref}, RMSD {min_rmsd:.5f}")

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
        for r in tqdm(pool.imap(align_to_references_mcs_hungarian_final, args_list), total=len(args_list), desc="Aligning tests"):
            if r is not None:
                results.append(r)

    return results

# ------------------------------
if __name__ == "__main__":
    mp.set_start_method("forkserver")
    results = align_folder_to_references(REFERENCE_FOLDER, TEST_FOLDER, OUTPUT_FOLDER)
    for test_name, ref_name, rmsd_val in results:
        print(f"{test_name} best aligned to {ref_name}, RMSD {rmsd_val:.5f}")
