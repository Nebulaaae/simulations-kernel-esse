import uproot
import pandas as pd
import numpy as np
import os
import subprocess
import sys
from scipy.ndimage import gaussian_filter

# --- CONFIGURATION DU KERNEL ESSE ---
PIXEL_SIZE = 0.48      # cm
MU_WATER = 0.15        # cm-1 (fKrnlMu0)
KRNL_SIZE = 64         # iNumXYoffs
WATERBOX_DEPTH = 30.0  # cm
OUTPUT_FOLDER = os.path.abspath("./output")
SIM_SCRIPT = os.path.abspath("./spect_main1.py")
SIM_SCRIPT_AIR = os.path.abspath("./spect_main2.py")

# --- CONFIGURATION DES SLICES ---
NB_SLICES = 5
RUNS_PER_SLICE = 7
Z_POSITIONS = np.linspace(-140, 140, NB_SLICES) 

CHECKPOINT_INTERVAL = 10
checkpoint_path = os.path.join(OUTPUT_FOLDER, "esse_kernels_checkpoint.npy")

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
        df_final['PostPosition_X'],
        df_final['PostPosition_Y'],
        bins=KRNL_SIZE,
        range=[[-limit, limit], [-limit, limit]],
        weights=df_final['ESSE_Weight']
    ) 
    # h = gaussian_filter(h, sigma=1.0) 
    return h, total_photons

# --- INITIALISATION ---
# final_kernels = np.zeros((NB_SLICES, KRNL_SIZE, KRNL_SIZE)) # todo : je crois que ça doit être stocké comme x, z, y (voir slide 39)
final_kernels = np.zeros((KRNL_SIZE, NB_SLICES, KRNL_SIZE))

# --- NORMALISATION DANS L'AIR ---
print("\n>>> Phase de normalisation : Simulation dans l'AIR")
nb_air_total = 0
AIR_RUNS = 2   # Plusieurs runs pour réduire la variance statistique

for a in range(AIR_RUNS):
    env = os.environ.copy()
    env["SOURCE_Z_POS"] = "0"
    subprocess.run([sys.executable, SIM_SCRIPT_AIR], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT_AIR))
    with uproot.open(os.path.join(OUTPUT_FOLDER, "spect.root")) as f:
        nb_air_total += f["peak208"].num_entries
    os.remove(os.path.join(OUTPUT_FOLDER, "spect.root"))

# Facteur de normalisation par run 
norm_factor = (nb_air_total / AIR_RUNS) * RUNS_PER_SLICE
print(f"Facteur de normalisation calculé : {norm_factor} photons")

# --- GÉNÉRATION DES KERNELS PAR PROFONDEUR ---
for s_idx, z_mm in enumerate(Z_POSITIONS):
    # depth : distance entre source et face de sortie (z=150mm)
    depth = (150.0 - z_mm) / 10.0
    print(f"\n=== SLICE {s_idx+1}/{NB_SLICES} (Z={z_mm}mm, Profondeur={depth:.2f}cm) ===")

    for r_idx in range(RUNS_PER_SLICE):
        total_run_idx = s_idx * RUNS_PER_SLICE + r_idx + 1
        
        env = os.environ.copy()
        env["SOURCE_Z_POS"] = str(z_mm)
        
        try:
            subprocess.run([sys.executable, SIM_SCRIPT], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT))
            
            h_slice, _ = filter_and_extract(depth)
            if h_slice is not None:
                final_kernels[:, s_idx, :] += h_slice
            
            # Nettoyage
            for f in ["spect.root", "phantom_scatters.root"]:
                p = os.path.join(OUTPUT_FOLDER, f)
                if os.path.exists(p): os.remove(p)

            if total_run_idx % CHECKPOINT_INTERVAL == 0:
                np.save(checkpoint_path, final_kernels)
                print(f"[Checkpoint] Run {total_run_idx} sauvegardé.")

        except Exception as e:
            print(f"Erreur Slice {s_idx} Run {r_idx}: {e}")

# --- NORMALISATION FINALE ET SAUVEGARDE ---
if norm_factor > 0:
    final_kernels /= (norm_factor)
    np.save(os.path.join(OUTPUT_FOLDER, "esse_kernels_3d.npy"), final_kernels)
    print("\nKernels ESSE 3D générés et normalisés avec succès.")
else:
    print("\nErreur : Normalisation impossible (norm_factor = 0)")