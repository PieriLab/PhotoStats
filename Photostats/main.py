import geometry_functions
from geometry_functions import xyz_to_rdkit_mol, numpy_geom,
from dim_red import reduce_features
from pathlib import Path
import tqdm

raw_spawns_folder = Path('../data/aligned_geometries/SeamStress/benzene/type2/spawn')
mecis = [ 
    Path('../data/aligned_geometries/SeamStress/benzene/type2/meci/Type1.xyz'),
    Path('../data/aligned_geometries/SeamStress/benzene/type2/meci/Type2.xyz'),
    Path('../../data/aligned_geometries/SeamStress/benzene/type2/meci/Type3.xyz'),
    
]


idx = []
geometries = []

smiles = []

for x in tqdm.tqdm(list(raw_spawns_folder.glob('*'))):
    mol = xyz_to_rdkit_mol(x,total_charge=0)
    smile = Chem.MolToSmiles(mol, canonical=True)
    geom = numpy_geom(mol)
    flattend_geom = geom.flatten()
    geometries.append(flattend_geom)
    idx.append(x.stem)
    smiles.append(smile)


for meci in mecis:
    print(meci)
    mol = xyz_to_rdkit_mol(meci,total_charge=0)
    smile = Chem.MolToSmiles(mol, canonical=True)

    geom = numpy_geom(mol)
    flattend_geom = geom.flatten()
    geometries.append(flattend_geom)
    smiles.append(smile)
    idx.append(meci.stem)

feature_vector = np.array(geometries)