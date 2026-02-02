from geometry_functions import xyz_to_rdkit_mol, numpy_geom
from tqdm import tqdm 
import check_identity
from rdkit import Chem
from collections import Counter
import pandas as pd 
from pathlib import Path 


meci_geom_folder = Path("/data/raw_geometries/meci/butadiene/")
class_file = Path('/data/meci_classification/butadiene/meci_labels.csv')

labels = {}
unique_geoms = {}

for x in tqdm.tqdm(meci_geom_folder.glob('*')):
    #print(x)
    test_mol = xyz_to_rdkit_mol(x, total_charge=0)
    smiles = Chem.MolToSmiles(test_mol, canonical=True)

    if smiles not in unique_geoms:
        unique_geoms[smiles] = [x]
        continue

    #Handle disconnected systems
    if "." in smiles:
        labels[x.stem] = unique_geoms[smiles][0].stem
        continue

    found_identical = False

    for template in unique_geoms[smiles]:
        template_mol = xyz_to_rdkit_mol(template, total_charge=0)

        identical, best_rmsd = check_identity(template_mol, test_mol)

        if identical:
            labels[x.stem] = template.stem
            found_identical = True
            break

    if not found_identical:
        unique_geoms[smiles].append(x)



label_counts = Counter(labels.values())

for label, count in label_counts.items():
    print(label, count)

df = pd.DataFrame(labels.items(), columns = ['idx','meci_type'])

df.to_csv(class_file, index=False)