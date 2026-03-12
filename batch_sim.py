import uproot
import pandas as pd
import numpy as np
import os
import subprocess
import sys

# --- CONFIGURATION DU KERNEL ESSE ---
PIXEL_SIZE = 0.48      # Taille de pixel (cm)
NB_RUN = 15         # iNumDists
MU_WATER = 0.15        # fKrnlMu0 (cm-1)
KRNL_SIZE = 64         # iNumXYoffs (Taille de la grille du kernel)
OUTPUT_FOLDER = os.path.abspath("./output")
SIM_SCRIPT = os.path.abspath("./spect_main1.py") 
WATERBOX_DEPTH = 30.0 # cm

def filter_and_extract(depth):
    scatter_path = os.path.join(OUTPUT_FOLDER, "phantom_scatters.root")
    spect_path = os.path.join(OUTPUT_FOLDER, "spect.root")
    
    if not (os.path.exists(scatter_path) and os.path.exists(spect_path)):
        return None, 0

    with uproot.open(spect_path) as f:
        tree = f["peak208"]
        df_det = tree.arrays(["EventID", "Weight"], library="pd")
        total_photons = tree.num_entries

    detected_ids = df_det[['EventID', 'Weight']].drop_duplicates('EventID')
    waterbox_path = f"{scatter_path}:Hits_Waterbox"
    extracted_points = []
    
    for chunk in uproot.iterate(waterbox_path, ["EventID", 'PostPosition_X', 'PostPosition_Y', 'PostPosition_Z', "ProcessDefinedStep"], library="pd"):
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
        return None, total_photons

    df_final = pd.concat(extracted_points)
    limit = (KRNL_SIZE * PIXEL_SIZE * 10) / 2
    
    h, _, _ = np.histogram2d(
        df_final['PostPosition_Y'], # Axe horizontal du schéma
        df_final['PostPosition_Z'], # Axe vertical du schéma (vers le détecteur)
        bins=KRNL_SIZE,
        range=[[-limit, limit], [-limit, limit]],
        weights=df_final['ESSE_Weight']
    ) 
    return h, total_photons

# --- BOUCLE PRINCIPALE ---
# Initialisation de la matrice (Slices, X, Y)
final_kernels = np.zeros((1, KRNL_SIZE, KRNL_SIZE))
CHECKPOINT_INTERVAL = 10
checkpoint_path = os.path.join(OUTPUT_FOLDER, "esse_kernels_checkpoint.npy")
stats_path = os.path.join(OUTPUT_FOLDER, "checkpoint_stats.txt")

for i in range(NB_RUN):
    print(f"\n>>> RUN {i+1}/{NB_RUN}")

    env = os.environ.copy()
    env["SOURCE_Z_POS"] = "0"
    depth = WATERBOX_DEPTH / 2 + float(env["SOURCE_Z_POS"]) / 10
    
    try:
        subprocess.run([sys.executable, SIM_SCRIPT], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT))
        
        slice_kernel, nb_photons = filter_and_extract(depth)
        # print(f"Photons totaux : {nb_photons}")
        # print(f"Points de diffusion utiles : {np.sum(slice_kernel) if slice_kernel is not None else 0}")
        if slice_kernel is not None:
            final_kernels[0,:, :] += slice_kernel
        os.remove(os.path.join(OUTPUT_FOLDER, "spect.root"))
        os.remove(os.path.join(OUTPUT_FOLDER, "phantom_scatters.root"))
        if (i+1) % CHECKPOINT_INTERVAL == 0:
            np.save(checkpoint_path, final_kernels)
            # Sauvegarde de l'état pour pouvoir reprendre la normalisation si crash
            with open(stats_path, "w") as f:
                f.write(f"last_completed_run:{i+1}\n")
            print(f"--- Checkpoint sauvegardé au run {i+1} ---")
                
    except Exception as e:
        print(f"Erreur au run {i}: {e}")

SIM_SCRIPT_AIR = os.path.abspath("./spect_main2.py")
subprocess.run([sys.executable, SIM_SCRIPT_AIR], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT_AIR))

with uproot.open(os.path.join(OUTPUT_FOLDER, "spect.root")) as f:
    tree = f["peak208"]
    nb_photons_air = tree.num_entries
print(f"\nPhotons détectés dans l'air : {nb_photons_air}")
nb_total_photons_air = nb_photons_air * NB_RUN

# Normalisation du kernel
if nb_total_photons_air > 0:
    final_kernels /= nb_total_photons_air
    print(f"Kernel normalisé par {nb_total_photons_air} photons.")
else:
    print("Aucun photon détecté dans l'air, normalisation impossible.")
os.remove(os.path.join(OUTPUT_FOLDER, "spect.root"))

# Sauvegarde au format NumPy
np.save(os.path.join(OUTPUT_FOLDER, "esse_kernels.npy"), final_kernels)
print("\nTable de kernels sauvegardée.")