from geometry_functions import xyz_to_rdkit_mol, numpy_geom, flatten_xyz_files
from plotting import plot_spawns_vs_mecis, plot_meci_types
from dim_red import reduce_features
from pathlib import Path
from tqdm import tqdm 

spawn_folder = Path('data/aligned_geometries/SeamStress/benzene/type2/spawn')
meci_folder =  Path('data/aligned_geometries/SeamStress/benzene/type2/meci/')
Molecule_name = 'Benzene'
reports_folder = Path(f"reports/figures/{Molecule_name}")
reports_folder.mkdir(exist_ok=True)
mecis = list(meci_folder.glob('*.xyz'))  
reduction_technique= "UMAP"



feature_vector, idx, smiles = flatten_xyz_files(spawn_folder, meci_folder)

reduced_space = reduce_features(feature_vector,reduction_technique=reduction_technique,n_components=2)

n_spawns = len(list(spawn_folder.glob('*')))

X_spawns = reduced_space[:n_spawns]
X_mecis  = reduced_space[n_spawns:]

smiles_spawns = smiles[:n_spawns]
smiles_mecis  = smiles[n_spawns:]
idx_mecis     = idx[n_spawns:]


plot_spawns_vs_mecis(
    Molecule_name,
    X_spawns, X_mecis,
    smiles_spawns, smiles_mecis,
    idx_mecis,
    reduction_technique=reduction_technique,
    style_file="/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/prl.mplstyle",
    save_path= reports_folder / f"{Molecule_name}_smiles.png"
)

meci_class_labels = "data/meci_classification/benzene/S1S0/meci_labels_humanlabels.csv"


plot_meci_types(
    Molecule_name,
    reduced_space,  
    idx,            
    mecis,          
    meci_class_labels,            
    reduction_technique=reduction_technique,
    save_path=reports_folder / f"{Molecule_name}_meci_types.png"
)
