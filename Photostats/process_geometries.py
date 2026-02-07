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

import os
import numpy as np
import pandas as pd
from ase.io import read

class GeometryDataset:
    def __init__(
        self,
        folder_path,
        meci_labels_csv=None,
        file_extension=".xyz",
        idx_key="idx",
        label_key="meci_type",
    ):
        self.folder_path = folder_path
        self.file_extension = file_extension

        self.structures = []
        self.names = []
        self.meci_labels = []

        # Load MECI labels if provided
        if meci_labels_csv is not None:
            df = pd.read_csv(meci_labels_csv)
            self._meci_label_dict = dict(zip(df[idx_key], df[label_key]))
        else:
            self._meci_label_dict = None

        self._load_files()

        if meci_labels_csv is not None:
            print(
                f"Loaded {len(self.structures)} labeled geometries "
                f"from {self.folder_path}"
            )

    def _load_files(self):
        for filename in sorted(os.listdir(self.folder_path)):
            if not filename.endswith(self.file_extension):
                continue

            name = os.path.splitext(filename)[0]

            # Skip if MECI labels provided and this geometry is unlabeled
            if self._meci_label_dict is not None:
                label = self._meci_label_dict.get(name, None)
                if label is None or pd.isna(label):
                    continue

            filepath = os.path.join(self.folder_path, filename)

            try:
                atoms = read(filepath)
                self.structures.append(atoms)
                self.names.append(name)

                if self._meci_label_dict is not None:
                    self.meci_labels.append(label)

            except Exception as e:
                print(f"Warning: could not read {filename}: {e}")

        if self._meci_label_dict is None:
            self.meci_labels = None

    def to_meci_dataframe(self):
        if self.meci_labels is None:
            raise RuntimeError("Dataset was created without MECI labels.")

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
