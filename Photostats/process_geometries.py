import os
from ase.io import read
import numpy as np
from dscribe.descriptors import SOAP
from dscribe.descriptors import MBTR
from scipy.spatial.distance import pdist, squareform
from ase import Atoms


import numpy as np
import os
import numpy as np
import pandas as pd
from ase.io import read

class GeometryDataset:
    def __init__(self, folder_path, file_extension=".xyz"):
        self.folder_path = folder_path
        self.file_extension = file_extension

        self.structures = []
        self.names = []
        self.meci_labels = None  # populated later

        self._load_files()

    def _load_files(self):
        for filename in sorted(os.listdir(self.folder_path)):
            if filename.endswith(self.file_extension):
                filepath = os.path.join(self.folder_path, filename)
                try:
                    atoms = read(filepath)
                    self.structures.append(atoms)
                    self.names.append(os.path.splitext(filename)[0])
                except Exception as e:
                    print(f"Warning: could not read {filename}: {e}")

    def attach_and_prune_meci_labels(self, meci_labels_csv, idx_key="idx", label_key="meci_type"):
        """
        Attach MECI labels to the dataset and prune unlabeled geometries.
        """
        df = pd.read_csv(meci_labels_csv)
        label_dict = dict(zip(df[idx_key], df[label_key]))

        labels_aligned = [
            label_dict.get(name, np.nan)
            for name in self.names
        ]

        mask = [not pd.isna(lbl) for lbl in labels_aligned]

        # Prune dataset in place
        self.structures = [
            s for s, keep in zip(self.structures, mask) if keep
        ]
        self.names = [
            n for n, keep in zip(self.names, mask) if keep
        ]
        self.meci_labels = [
            lbl for lbl, keep in zip(labels_aligned, mask) if keep
        ]

        print(
            f"Dataset pruned: kept {len(self.names)} / {len(mask)} geometries "
            f"with MECI labels"
        )

    def to_meci_dataframe(self):
        """
        Return MECI labels as a DataFrame aligned with the dataset.
        """
        if self.meci_labels is None:
            raise RuntimeError("MECI labels not attached yet.")

        return pd.DataFrame({
            "idx": self.names,
            "meci_label": self.meci_labels
        })

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        if self.meci_labels is None:
            return self.structures[idx], self.names[idx]
        return self.structures[idx], self.names[idx], self.meci_labels[idx]



def generate_SOAP(atoms, r_cut=5.0, n_max=8, l_max=6, average = 'inner'):
    """
    Generate a SOAP feature vector for a single ASE Atoms object.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The structure to generate SOAP for.
    species : list, optional
        List of chemical species in your dataset. If None, it will be inferred.
    rcut : float
        Cutoff radius for SOAP.
    nmax : int
        Radial basis functions.
    lmax : int
        Angular momentum basis functions.
    
    Returns
    -------
    np.ndarray
        SOAP feature vector (flattened).
    """

   
    species = list(set(atoms.get_chemical_symbols()))
    
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
        sparse=False
        

    )
    
    # SOAP expects a list of Atoms objects
    feature = soap.create([atoms])
    return feature.flatten()



def generate_inv_eigenval(atoms):
    """
    Generate a feature vector from the inverse eigenvalues of the distance matrix.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to compute features for.
    
    Returns
    -------
    np.ndarray
        Inverse eigenvalue feature vector.
    """
    positions = atoms.get_positions()  
    
    dist_matrix = squareform(pdist(positions))  

    eps = 1e-8
    inv_dist_matrix = 1 / (dist_matrix + eps)
    
    inv_eigenvals = np.linalg.eigvals(inv_dist_matrix)  
    
    inv_eigenvals = np.sort(inv_eigenvals)[::-1]  # descending order
    
    return inv_eigenvals


def generate_inverse_dist_matrix(atoms):
    """
    Generate a feature vector from the inverse distance matrix.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to compute features for.
    
    Returns
    -------
    np.ndarray
        Inverse eigenvalue feature vector.
    """
    positions = atoms.get_positions()  
    
    dist_matrix = squareform(pdist(positions))  
    flattened_matrix = dist_matrix.flatten()
    
    return flattened_matrix

def generate_MBTR(atoms,  normalization = 'none' ):
    """
    Generate MBTR feature vector for a single ASE Atoms object.
    
    Parameters
    ----------
    atoms : ase.Atoms
        The structure to convert.
    species : list, optional
        List of all chemical species in your dataset. If None, inferred from atoms.
    k1_range : tuple
        Min and max for k1 (1-body term) distances, optional.
    k2_range : tuple
        Min and max for k2 (2-body term) distances, optional.
    
    Returns
    -------
    np.ndarray
        Flattened MBTR feature vector.
    """
 
    species = list(set(atoms.get_chemical_symbols()))
    
    mbtr = MBTR(
                geometry={"function": "distance"},
                grid={"min": 0.5, "max": 5.0, "sigma": 0.1, "n": 50},
                weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
                normalization=normalization,
                species=species,
                periodic=False
)
    
    feature = mbtr.create([atoms])
    
    return feature


def flatten_cartesian(atoms):
    positions = atoms.get_positions() 
    return positions.flatten() 
