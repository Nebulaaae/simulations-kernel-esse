import uproot
import pandas as pd
import numpy as np
import os
import subprocess
import sys

# --- CONFIGURATION DU KERNEL ESSE ---
PIXEL_SIZE = 0.48      # Taille de pixel (cm)
NB_SLICES = 15         # iNumDists
MU_WATER = 0.15        # fKrnlMu0 (cm-1)
KRNL_SIZE = 64         # iNumXYoffs (Taille de la grille du kernel)
OUTPUT_FOLDER = os.path.abspath("./output")
# Chemin absolu vers le script de simulation
SIM_SCRIPT = os.path.abspath("../spect_main1.py") 

# def filter_and_extract(depth):
#     scatter_path = os.path.join(OUTPUT_FOLDER, "phantom_scatters.root")
#     spect_path = os.path.join(OUTPUT_FOLDER, "spect.root")
    
#     if not (os.path.exists(scatter_path) and os.path.exists(spect_path)):
#         return None

#     # 1. IDs des photons ayant diffusé (Compton)
#     with uproot.open(scatter_path) as f:
#         df_hits = f["Hits_Waterbox"].arrays(["EventID", "ProcessDefinedStep"], library="pd")
#         df_hits['ProcessDefinedStep'] = df_hits['ProcessDefinedStep'].astype(str)
#         scattered_ids = df_hits[df_hits['ProcessDefinedStep'].str.contains('compt')]['EventID'].unique()

#     # 2. Position d'impact sur le détecteur (PSF)
#     with uproot.open(spect_path) as f:
#         df_det = f["peak208"].arrays(["EventID", "PostPosition_X", "PostPosition_Y", "Weight"], library="pd")
#         useful = df_det[df_det['EventID'].isin(scattered_ids)].copy()

#     if not useful.empty:
#         # Poids ESSE : Source Efficace
        
        
#         # 3. Binning : Transformation des points en image 2D (le Kernel)
#         # On centre le kernel sur (0,0). Range à adapter selon la taille du détecteur.
#         limit = (KRNL_SIZE * PIXEL_SIZE * 10) / 2 # en mm
#         h, xedges, yedges = np.histogram2d(
#             useful['PostPosition_X'], 
#             useful['PostPosition_Y'], 
#             bins=KRNL_SIZE,
#             range=[[-limit, limit], [-limit, limit]],
#             weights=useful['ESSE_Weight']
#         )
#         print(f"Kernel avec {len(useful)} points utiles.")
#         return h
#     return None

# def filter_and_extract():
#     """Extrait les scatters utiles du run actuel"""
#     # 1. Charger les IDs du détecteur (Peak 208)
#     with uproot.open(f"{OUTPUT_FOLDER}/spect.root") as f:
#         df_det = f["peak208"].arrays(["EventID", "Weight"], library="pd")
#         detected_ids = df_det.drop_duplicates(['EventID'])

#     # 2. Filtrer le Phantom
#     waterbox_path = f"{OUTPUT_FOLDER}/phantom_scatters.root:Hits_Waterbox"
#     extracted_points = []
    
#     # Lecture par chunk pour la sécurité
#     for chunk in uproot.iterate(waterbox_path, ["EventID", "PostPosition_X", "PostPosition_Y", "PostPosition_Z", "ProcessDefinedStep"], library="pd"):
#         chunk['ProcessDefinedStep'] = chunk['ProcessDefinedStep'].astype(str)
#         mask = (chunk['ProcessDefinedStep'].str.contains('compt')) & (chunk['EventID'].isin(detected_ids['EventID']))
#         useful = chunk[mask]
#         if not useful.empty:
#             merged = useful.merge(detected_ids, on='EventID', how='inner')
#             if not merged.empty:
#                 merged['ESSE_Weight'] = useful['Weight'] * np.exp(MU_WATER * depth)
#                 extracted_points.append(merged.groupby('EventID').tail(1))

#     if extracted_points:
#         return pd.concat(extracted_points)
#     return pd.DataFrame()

def filter_and_extract(depth):
    print("extraction")
    scatter_path = os.path.join(OUTPUT_FOLDER, "phantom_scatters.root")
    spect_path = os.path.join(OUTPUT_FOLDER, "spect.root")
    
    if not (os.path.exists(scatter_path) and os.path.exists(spect_path)):
        return None, 0

    with uproot.open(spect_path) as f:
        tree = f["peak208"]
        df_det = tree.arrays(["EventID", "Weight"], library="pd")
        total_photons = tree.num_entries

    detected_ids = df_det[['EventID', 'PostPosition_X', 'PostPosition_Y', 'PostPosition_Z', 'Weight']].drop_duplicates('EventID')
    waterbox_path = f"{scatter_path}:Hits_Waterbox"
    extracted_points = []
    
    for chunk in uproot.iterate(waterbox_path, ["EventID", "ProcessDefinedStep"], library="pd"):
        chunk['ProcessDefinedStep'] = chunk['ProcessDefinedStep'].astype(str)
        mask = (chunk['ProcessDefinedStep'].str.contains('compt')) & (chunk['EventID'].isin(detected_ids['EventID']))
        useful = chunk[mask]
        
        if not useful.empty:
            last_hits = useful.groupby('EventID').tail(1)
            merged = last_hits.merge(df_det, on='EventID', how='inner')
            if not merged.empty:
                merged['ESSE_Weight'] = merged['Weight'] * np.exp(MU_WATER * depth)
                extracted_points.append(merged)

    if not extracted_points:
        print("Kernel sans points de diffusions détectés")
        return None, total_photons

    df_final = pd.concat(extracted_points)
    limit = (KRNL_SIZE * PIXEL_SIZE * 10) / 2
    
    h, _, _ = np.histogram2d(
        df_final['PostPosition_X'], 
        df_final['PostPosition_Y'], 
        bins=KRNL_SIZE,
        range=[[-limit, limit], [-limit, limit]],
        weights=df_final['ESSE_Weight']
    )

    print(f"Kernel avec {len(df_final)} points de diffusions détectés sur {total_photons} photons.")
    
    return h, total_photons

# --- BOUCLE PRINCIPALE ---
# Initialisation de la matrice (Slices, X, Y)
final_kernels = np.zeros((NB_SLICES, KRNL_SIZE, KRNL_SIZE))

for i in range(NB_SLICES):
    depth = (i + 0.5) * PIXEL_SIZE
    z_source_mm = -depth * 10
    
    print(f"\n>>> RUN {i+1}/{NB_SLICES} | Depth: {depth:.2f} cm")

    env = os.environ.copy()
    env["SOURCE_Z_POS"] = str(z_source_mm)
    
    try:
        # Exécution avec le chemin absolu du script
        subprocess.run([sys.executable, SIM_SCRIPT], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT))
        
        slice_kernel, nb_photons = filter_and_extract(depth)
        print(f"Photons totaux : {nb_photons}")
        print(f"Points de diffusion utiles : {np.sum(slice_kernel) if slice_kernel is not None else 0}")
        if slice_kernel is not None:
            final_kernels[i, :, :] = slice_kernel
            print(f"Slice {i}.")
            
    except Exception as e:
        print(f"Erreur au run {i}: {e}")

# Sauvegarde au format NumPy (facilement convertible en .krnl ou Interfile)
np.save("esse_kernels_3d.npy", final_kernels)
print("\nTable de kernels sauvegardée.")