
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist, squareform
import numpy as np 

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




def align_and_rmsd_numpy(X, Y, return_aligned=False):
    X_center = X.mean(axis=0)
    Y_center = Y.mean(axis=0)
    Xc = X - X_center
    Yc = Y - Y_center

    rot, _ = R.align_vectors(Xc, Yc)
    Y_aligned = rot.apply(Yc) + X_center
    rmsd = np.sqrt(np.mean(np.sum((X - Y_aligned)**2, axis=1)))

    if return_aligned:
        return rmsd, Y_aligned
    else:
        return rmsd
    




def update_mol_coordinates_copy(mol, new_coords):
    """
    Return a copy of an RDKit molecule with updated 3D coordinates.
    
    Parameters:
        mol: RDKit Mol object (must have same number of atoms as new_coords)
        new_coords: (N,3) numpy array of new coordinates
    
    Returns:
        new_mol: RDKit Mol object copy with updated coordinates
    """
    if new_coords.shape[0] != mol.GetNumAtoms():
        raise ValueError("Number of coordinates must match number of atoms")
    
    # --- Make a true copy ---
    new_mol = Chem.Mol(mol)  # copy molecule
    # Remove old conformers
    for conf_id in [c.GetId() for c in new_mol.GetConformers()]:
        new_mol.RemoveConformer(conf_id)
    
    # --- Add new conformer ---
    conf = Chem.Conformer(new_mol.GetNumAtoms())
    for i, pos in enumerate(new_coords):
        conf.SetAtomPosition(i, pos)
    new_mol.AddConformer(conf)
    
    return new_mol



def reflect_axis(geometry, reflection): 
    reflected_geom = geometry * reflection 
    return reflected_geom


REFLECTIONS = np.array(
    [
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, -1],
        [-1, -1, -1],
    ]
)

def numpy_geom(mol):
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])




