import uproot
import numpy as np
import pandas as pd
import os
import SimpleITK as sitk

OUTPUT_FOLDER = os.path.abspath("./nema_simulation_v1")
PIXEL_SIZE = 4.4 
IMG_SIZE = 128

def extract_ground_truth_images():
    scatter_path = os.path.join(OUTPUT_FOLDER, "phantom_scatters_gt.root")
    spect_path = os.path.join(OUTPUT_FOLDER, "spect_hits.root")
    
    if not (os.path.exists(scatter_path) and os.path.exists(spect_path)):
        print(f"Erreur : Fichiers introuvables.")
        return

    print("Lecture des détections cristal...")
    with uproot.open(spect_path) as f:
        crystal_key = [k for k in f.keys() if "Hits" in k and "crystal" in k.lower()][0]
        df_det = f[crystal_key].arrays(["EventID", "PostPosition_X", "PostPosition_Y"], library="pd")
        detected_ids = df_det[['EventID']].drop_duplicates('EventID')

    print(f"Photons détectés dans le cristal : {len(df_det)}")

    print("Filtrage des scatters par itération...")
    extracted_scatter_points = []
    phantom_tree_path = f"{scatter_path}:Hits_Phantom"
    
    for chunk in uproot.iterate(phantom_tree_path, 
                                ["EventID", "PostPosition_X", "PostPosition_Y", "ProcessDefinedStep"], 
                                step_size="100MB", library="pd"):
        
        chunk['ProcessDefinedStep'] = chunk['ProcessDefinedStep'].astype(str)
        
        mask = (chunk['ProcessDefinedStep'].str.contains('compt')) & \
               (chunk['EventID'].isin(detected_ids['EventID']))
        
        useful = chunk[mask]
        
        if not useful.empty:
            last_hits = useful.groupby('EventID').tail(1)
            merged = last_hits.merge(df_det, on='EventID', how='inner', suffixes=('_phantom', '_crystal'))
            if not merged.empty:
                extracted_scatter_points.append(merged)

    if not extracted_scatter_points:
        print("Aucun scatter trouvé avec ces critères.")
        return

    df_scatter_final = pd.concat(extracted_scatter_points)
    scatter_ids = set(df_scatter_final['EventID'])


    df_primary_final = df_det[~df_det['EventID'].isin(scatter_ids)]

    limit = (IMG_SIZE * PIXEL_SIZE) / 2
    range_img = [[-limit, limit], [-limit, limit]]

    h_scatter, _, _ = np.histogram2d(
        df_scatter_final['PostPosition_X_crystal'], df_scatter_final['PostPosition_Y_crystal'],
        bins=IMG_SIZE, range=range_img
    )

    h_primary, _, _ = np.histogram2d(
        df_primary_final['PostPosition_X'], df_primary_final['PostPosition_Y'],
        bins=IMG_SIZE, range=range_img
    )

    # Sauvegarde
    save_mhd(h_primary, "primary_reference.mhd")
    save_mhd(h_scatter, "scatter_reference.mhd")
    
    print("-" * 30)
    print(f"Extraction terminée.")
    print(f"Primaires réels : {len(df_primary_final)}")
    print(f"Scatters réels   : {len(df_scatter_final)}")
    print("-" * 30)

def save_mhd(data, filename):
    img = sitk.GetImageFromArray(data.T)
    img.SetSpacing([PIXEL_SIZE, PIXEL_SIZE, 1.0])
    sitk.WriteImage(img, os.path.join(OUTPUT_FOLDER, filename))

if __name__ == "__main__":
    extract_ground_truth_images()