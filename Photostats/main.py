import geometry_functions
from geometry_functions import xyz_to_rdkit_mol, numpy_geom, flatten_xyz_files
from plotting import plot_spawns_vs_mecis, plot_meci_types
from dim_red import reduce_features
from pathlib import Path
from tqdm import tqdm 

spawn_folder = Path('data/aligned_geometries/SeamStress/benzene/type2/spawn')
mecis = [ 
    Path('data/aligned_geometries/SeamStress/benzene/type2/meci/Type1.xyz'),
    Path('data/aligned_geometries/SeamStress/benzene/type2/meci/Type2.xyz'),
    Path('data/aligned_geometries/SeamStress/benzene/type2/meci/Type3.xyz'),
    
]
reduction_technique= "UMAP"



feature_vector, idx, smiles = flatten_xyz_files(spawn_folder, mecis)

reduced_space = reduce_features(feature_vector,reduction_technique=reduction_technique,n_components=2)

n_spawns = len(list(spawn_folder.glob('*')))

X_spawns = reduced_space[:n_spawns]
X_mecis  = reduced_space[n_spawns:]

smiles_spawns = smiles[:n_spawns]
smiles_mecis  = smiles[n_spawns:]
idx_mecis     = idx[n_spawns:]


plot_spawns_vs_mecis(
    X_spawns, X_mecis,
    smiles_spawns, smiles_mecis,
    idx_mecis,
    reduction_technique=reduction_technique,
    style_file="/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/prl.mplstyle"
)

# Path to your MECI classification CSV
csv_file = "data/meci_classification/benzene/S1S0/meci_labels_humanlabels.csv"

n_star = len(mecis)

plot_meci_types(
    reduced_space,  
    idx,            
    mecis,          
    csv_file,       
    n_star,         
    reduction_technique=reduction_technique  
)
