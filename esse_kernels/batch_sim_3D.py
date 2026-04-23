import uproot
import pandas as pd
import numpy as np
import os
import subprocess
import sys
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# --- CONFIGURATION DU KERNEL ESSE ---
PIXEL_SIZE = 0.48      # cm
MU_WATER = 0.135        # cm-1 (fKrnlMu0)
KRNL_SIZE = 64         # iNumXYoffs
WATERBOX_DEPTH = 30.0  # cm
OUTPUT_FOLDER = os.path.abspath("./output")
SIM_SCRIPT = os.path.abspath("./spect_slab.py")
SIM_SCRIPT_AIR = os.path.abspath("./spect_air.py")

def get_mu_water(energy_kev):
    """
    Interpolation simplifiée du mu de l'eau (cm-1).
    todo : remplacer par une vraie fonction d'interpolation à partir de données réelles
    """

    energies = np.array([50, 100, 150, 208, 250, 322])
    mus = np.array([0.22, 0.17, 0.15, 0.135, 0.125, 0.118])
    return np.interp(energy_kev, energies, mus)

# --- CONFIGURATION DES SLICES ---
NB_SLICES = 20
RUNS_PER_SLICE = 5
Z_POSITIONS = np.linspace(-140, 140, NB_SLICES) 

CHECKPOINT_INTERVAL = 10
checkpoint_path = os.path.join(OUTPUT_FOLDER, "esse_kernels_checkpoint.npy.tmp.npy")
print(f"Checkpoint path: {checkpoint_path}")
print(os._exists(checkpoint_path))

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
    print(f"Total photons detected: {total_photons}, Unique EventIDs: {len(detected_ids)}")
    waterbox_path = f"{scatter_path}:Hits_Waterbox"
    extracted_points = []
    
    for chunk in uproot.iterate(waterbox_path, ["EventID", 'PostPosition_X', 'PostPosition_Y', 
                                                'PostPosition_Z', "ProcessDefinedStep", "KineticEnergy"], library="pd"):
        chunk['ProcessDefinedStep'] = chunk['ProcessDefinedStep'].astype(str)
        mask = (chunk['ProcessDefinedStep'].str.contains('compt')) & (chunk['EventID'].isin(detected_ids['EventID']))#& (np.abs(chunk['PostDirection_Z']) > 0.999)
        useful = chunk[mask]
        
        if not useful.empty:
            last_hits = useful.groupby('EventID').tail(1)
            merged = last_hits.merge(df_det, on='EventID', how='inner')
            if not merged.empty:
                merged['mu_i'] = get_mu_water(merged['KineticEnergy'] * 1000)
                
                merged['delta_mu_weighted'] = (merged['mu_i'] - MU_WATER) * merged['Weight']
                merged['ESSE_Weight'] = merged['Weight'] * np.exp(MU_WATER * (150.0 - merged['PostPosition_Z']) / 10.0)
                
                extracted_points.append(merged)

    if not extracted_points:
        return None, None, total_photons

    df_final = pd.concat(extracted_points)
    limit = (KRNL_SIZE * PIXEL_SIZE * 10) / 2
    
    # Dénominateur : Somme des poids de diffusion (le futur pfKrnl)
    h_weight_sum, _, _ = np.histogram2d(
        df_final['PostPosition_X'], df_final['PostPosition_Y'],
        bins=KRNL_SIZE, range=[[-limit, limit], [-limit, limit]],
        weights=df_final['ESSE_Weight']
    ) 

    # Numérateur : Somme des (delta_mu * poids)
    h_delta_mu_sum, _, _ = np.histogram2d(
        df_final['PostPosition_X'], df_final['PostPosition_Y'],
        bins=KRNL_SIZE, range=[[-limit, limit], [-limit, limit]],
        weights=df_final['delta_mu_weighted']
    )

    return h_weight_sum, h_delta_mu_sum, total_photons

# --- INITIALISATION ---
# final_kernels = np.zeros((NB_SLICES, KRNL_SIZE, KRNL_SIZE)) # todo : je crois que ça doit être stocké comme x, z, y (voir slide 39)
final_kernels = np.zeros((KRNL_SIZE, NB_SLICES, KRNL_SIZE))
amu_kernels_accumulation = np.zeros((KRNL_SIZE, NB_SLICES, KRNL_SIZE))
final_amu_kernels = np.zeros((KRNL_SIZE, NB_SLICES, KRNL_SIZE))
# --- INITIALISATION & REPRISE ---
final_kernels = np.zeros((KRNL_SIZE, NB_SLICES, KRNL_SIZE))
amu_kernels_accumulation = np.zeros((KRNL_SIZE, NB_SLICES, KRNL_SIZE))
start_slice = 0
start_run = 0

if os.path.exists(checkpoint_path):
    print(f">>> Chargement du checkpoint : {checkpoint_path}")
    cp = np.load(checkpoint_path, allow_pickle=True).item()
    
    final_kernels = cp['kernels']
    amu_kernels_accumulation = cp['amu_accum']
    norm_factor = cp['norm_factor']
    
    # On reprend à la slice suivante si le dernier run de la slice était fini, 
    # ou on gère la reprise au sein de la slice.
    last_run_idx = cp['last_run']
    last_slice_idx = cp['last_slice']
    
    if last_run_idx == RUNS_PER_SLICE - 1:
        start_slice = last_slice_idx + 1
        start_run = 0
    else:
        start_slice = last_slice_idx
        start_run = last_run_idx + 1
    print(f"Reprise à partir de : Slice {start_slice + 1}, Run {start_run + 1}")
else:
    print(">>> Aucun checkpoint trouvé. Démarrage d'une nouvelle simulation.")
    # Exécuter ici votre logique de calcul de norm_factor (Phase de normalisation AIR)

    # --- NORMALISATION DANS L'AIR ---
    print("\n>>> Phase de normalisation : Simulation dans l'AIR")
    nb_air_total = 0
    AIR_RUNS = 1   # Plusieurs runs pour réduire la variance statistique

    for a in range(AIR_RUNS):
        env = os.environ.copy()
        env["SOURCE_Z_POS"] = "0"
        subprocess.run([sys.executable, SIM_SCRIPT_AIR], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT_AIR))
        spect_air_path = os.path.join(OUTPUT_FOLDER, "spect.root")
        with uproot.open(spect_air_path) as f:
            tree = f["peak208"]
            # directions_z = tree.arrays(["PostDirection_Z"], library="pd")
            # count_filtered = (np.abs(directions_z['PostDirection_Z']) > 0.999).sum()
            # nb_air_total += count_filtered
            nb_air_total += tree.num_entries
            # print(f"  Run AIR {a+1}: {count_filtered} photons perpendiculaires retenus.")

        if os.path.exists(spect_air_path):
            os.remove(spect_air_path)

    # Facteur de normalisation par run 
    norm_factor = (nb_air_total / AIR_RUNS) * RUNS_PER_SLICE
    print(f"Facteur de normalisation calculé : {norm_factor} photons")

# --- GÉNÉRATION DES KERNELS PAR PROFONDEUR ---
for s_idx in range(start_slice, NB_SLICES):
    z_mm = Z_POSITIONS[s_idx]
    depth = (150.0 - z_mm) / 10.0
    
    print(f"\n=== SLICE {s_idx+1}/{NB_SLICES} (Z={z_mm}mm, Profondeur={depth:.2f}cm) ===")
    
    # Déterminer le run de départ pour cette slice (start_run seulement pour la première slice reprise)
    current_start_run = start_run if s_idx == start_slice else 0
    
    for r_idx in range(current_start_run, RUNS_PER_SLICE):
        total_run_idx = s_idx * RUNS_PER_SLICE + r_idx + 1
        
        env = os.environ.copy()
        env["SOURCE_Z_POS"] = str(z_mm)

        try:
            # Exécution de la simulation Gate/Python
            subprocess.run([sys.executable, SIM_SCRIPT], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT))
            
            # Extraction des données du ROOT
            h_slice, h_delta_num, _ = filter_and_extract(depth)
            
            if h_slice is not None:
                # Accumulation dans les matrices 3D
                final_kernels[:, s_idx, :] += h_slice
                amu_kernels_accumulation[:, s_idx, :] += h_delta_num

            # Nettoyage immédiat des fichiers volumineux
            for f in ["spect.root", "phantom_scatters.root"]:
                p = os.path.join(OUTPUT_FOLDER, f)
                if os.path.exists(p): os.remove(p)

            # --- SAUVEGARDE DU CHECKPOINT ---
            if total_run_idx % CHECKPOINT_INTERVAL == 0:
                checkpoint_data = {
                    'last_slice': s_idx,
                    'last_run': r_idx,
                    'kernels': final_kernels,
                    'amu_accum': amu_kernels_accumulation,
                    'norm_factor': norm_factor
                }
                
                # Utilisation d'un file object pour éviter l'ajout automatique de .npy par numpy
                temp_path = checkpoint_path + ".tmp"
                with open(temp_path, 'wb') as f:
                    np.save(f, checkpoint_data)
                
                os.replace(temp_path, checkpoint_path)
                print(f"[Checkpoint] Run {total_run_idx} (Slice {s_idx+1}, Run {r_idx+1}) sauvegardé.")

        except Exception as e:
            print(f"!!! Erreur critique Slice {s_idx} Run {r_idx}: {e}")
            # Optionnel : raise e si vous voulez stopper net en cas de problème système

# --- POST-TRAITEMENT GLOBAL (Une seule fois après toutes les slices) ---
print("\n>>> Calcul final des kernels amu (a_mu)...")
final_amu_kernels = np.divide(
    amu_kernels_accumulation, 
    final_kernels, 
    out=np.zeros_like(amu_kernels_accumulation), 
    where=final_kernels != 0
)

# 3. GÉNÉRATION DE L'IMAGE DE CONTRÔLE
print("\n>>> Génération de l'image de diagnostic : diag_esse.png")
plt.figure(figsize=(15, 5))
# Affichage du Kernel de base (Log scale pour voir les queues de diffusion)
plt.subplot(1, 3, 1)
plt.imshow(final_kernels[:, NB_SLICES//2, :], cmap='hot')
plt.title("Kernel Diffusion (Poids)")
plt.colorbar()

# Affichage du Numérateur (pour voir s'il y a de l'info avant division)
plt.subplot(1, 3, 2)
plt.imshow(amu_kernels_accumulation[:, NB_SLICES//2, :], cmap='magma')
plt.title("Numérateur Delta Mu")
plt.colorbar()

# Affichage du Ratio Final (amu)
plt.subplot(1, 3, 3)
plt.imshow(final_amu_kernels[:, NB_SLICES//2, :], cmap='viridis')
plt.title("Kernel Delta Mu Final")
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "diag_esse.png"))
print(f"Image sauvegardée dans : {os.path.join(OUTPUT_FOLDER, 'diag_esse.png')}")

# --- NORMALISATION FINALE ET SAUVEGARDE ---
if norm_factor > 0:
    final_kernels /= (norm_factor)
    np.save(os.path.join(OUTPUT_FOLDER, "esse_kernels_3d.npy"), final_kernels)
    np.save(os.path.join(OUTPUT_FOLDER, "esse_amu_kernels_3d.npy"), final_amu_kernels)
    print("\nKernels ESSE 3D générés et normalisés avec succès.")
else:
    print("\nErreur : Normalisation impossible (norm_factor = 0)")