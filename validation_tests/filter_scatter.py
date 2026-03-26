import uproot
import numpy as np
import pandas as pd
import os
import SimpleITK as sitk

OUTPUT_FOLDER = "./nema_simulation_v1"
PIXEL_SIZE = 4.4  # mm (GE Discovery standard)
IMG_SIZE = 128    # pixels

def extract_ground_truth_images():
    scatter_path = os.path.join(OUTPUT_FOLDER, "phantom_scatters_gt.root")
    spect_path = os.path.join(OUTPUT_FOLDER, "spect_hits.root")
    
    if not (os.path.exists(scatter_path) and os.path.exists(spect_path)):
        print("Erreur : Fichiers ROOT introuvables.")
        return

    # 1. Identifier TOUS les EventID qui ont diffusé dans le fantôme
    print("Identification des scatters dans le fantôme...")
    with uproot.open(scatter_path) as f:
        # On cherche l'arbre qui contient les hits du fantôme
        tree_name = [k for k in f.keys() if "Hits_Phantom" in k][0]
        # On ne récupère que les EventID des diffusions Compton
        # On utilise un set pour une recherche ultra-rapide (O(1))
        scatter_tree = f[tree_name]
        processes = scatter_tree.arrays(["EventID", "ProcessDefinedStep"], library="pd")
        processes['ProcessDefinedStep'] = processes['ProcessDefinedStep'].astype(str)
        
        scatter_event_ids = set(processes[processes['ProcessDefinedStep'].str.contains('compt')]['EventID'])

    # 2. Charger les détections dans le cristal
    print("Lecture des détections cristal...")
    with uproot.open(spect_path) as f:
        tree_name = [k for k in f.keys() if "Hits" in k and "crystal" in k.lower()][0]
        df_det = f[tree_name].arrays(["EventID", "PostPosition_X", "PostPosition_Y", "Weight"], library="pd")

    # 3. Séparation des données
    # Masque booléen : True si l'EventID est dans la liste des scatters
    is_scatter = df_det['EventID'].isin(scatter_event_ids)
    
    df_scatter = df_det[is_scatter]
    df_primary = df_det[~is_scatter]

    # 4. Création des histogrammes (Images)
    # On définit les limites pour centrer l'image
    limit = (IMG_SIZE * PIXEL_SIZE) / 2
    bins = [IMG_SIZE, IMG_SIZE]
    range_img = [[-limit, limit], [-limit, limit]]

    img_scatter, _, _ = np.histogram2d(
        df_scatter['PostPosition_X'], df_scatter['PostPosition_Y'],
        bins=bins, range=range_img, weights=df_scatter['Weight']
    )

    img_primary, _, _ = np.histogram2d(
        df_primary['PostPosition_X'], df_primary['PostPosition_Y'],
        bins=bins, range=range_img, weights=df_primary['Weight']
    )

    # 5. Sauvegarde en .mhd pour analyse (ou .npy)
    save_mhd(img_primary, "primary_reference.mhd")
    save_mhd(img_scatter, "scatter_reference.mhd")
    
    print(f"Extraction terminée.")
    print(f"Photons primaires : {len(df_primary)}")
    print(f"Photons diffusés : {len(df_scatter)}")

def save_mhd(data, filename):
    img = sitk.GetImageFromArray(data.T)
    img.SetSpacing([PIXEL_SIZE, PIXEL_SIZE, 1.0])
    sitk.WriteImage(img, os.path.join(OUTPUT_FOLDER, filename))

if __name__ == "__main__":
    extract_ground_truth_images()