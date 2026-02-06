import argparse
from tqdm import tqdm
from feature_comparison import create_meci_df_from_dataset
from process_geometries import (
    GeometryDataset,
    generate_SOAP,
    generate_inv_eigenval,
    generate_inverse_dist_matrix,
    generate_MBTR,
    flatten_cartesian
)

def main():
    #parser = argparse.ArgumentParser(description="Process XYZ geometries and MECI labels")
    #parser.add_argument("folder", type=str, help="Path to folder containing XYZ files")
    #parser.add_argument("meci_class_csv", type=str, help="Path to CSV of MECI labels")

    #args = parser.parse_args()





    #dataset = GeometryDataset(args.folder)
    dataset = GeometryDataset('/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/aligned_geometries/SeamStress/benzene/type2/spawn')
    print(f"Loaded {len(dataset)} structures:\n")




    #meci_df = create_meci_df_from_dataset(args.meci_class_csv)

    meci_df = create_meci_df_from_dataset( dataset, '/Users/connerbaucom/Desktop/Pieri/CTG/dim_red_comp/PhotoStats/data/meci_classification/benzene/S1S0/meci_labels_humanlabels.csv')
    print(f"Loaded {len(meci_df)} class labels")
    print(meci_df)



    feature_list = []

    for atoms, name in tqdm(
        zip(dataset.structures, dataset.names),
        total=len(dataset),
        desc="Generating features"
    ):
        features = {
            "name": name,
            "SOAP": generate_SOAP(atoms),
            "inv_eigenval": generate_inv_eigenval(atoms),
            "inverse_dist_matrix": generate_inverse_dist_matrix(atoms),
            "MBTR": generate_MBTR(atoms),
            "flatten_cartesian": flatten_cartesian(atoms),
        }
        feature_list.append(features)


if __name__ == "__main__":
    main()
