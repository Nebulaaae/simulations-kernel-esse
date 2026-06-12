import uproot
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import matplotlib.pyplot as plt

# --- CONFIGURATION DU KERNEL ESSE ---
PIXEL_SIZE = 0.48      # cm
MU_WATER = 0.135       # cm-1 (fKrnlMu0 à 140/208 keV)
KRNL_SIZE_XY = 64      # Taille spatiale X et Y
KRNL_SIZE_Z = 64       # Taille spatiale Z (profondeur de diffusion relative)
WATERBOX_EXIT_Z = 150.0 # mm (Position de la face de sortie de la cuve d'eau)

OUTPUT_FOLDER = os.path.abspath("./output")
SIM_SCRIPT = os.path.abspath("./spect_slab.py")
SIM_SCRIPT_AIR = os.path.abspath("./spect_air.py")

def get_mu_water(energy_kev):
    """Interpolation simplifiée du mu de l'eau (cm-1)."""
    energies = np.array([50, 100, 150, 208, 250, 322])
    mus = np.array([0.22, 0.17, 0.15, 0.135, 0.125, 0.118])
    return np.interp(energy_kev, energies, mus)

# --- CONFIGURATION DE LA SIMULATION (SOURCE FIXE) ---
TOTAL_RUNS = 80       # Plus on a de runs sur une seule position, moins il y a de bruit
SOURCE_Z_MM = 0.0     # On simule uniquement au centre ! (Spatially invariant)

CHECKPOINT_INTERVAL = 2
checkpoint_path = os.path.join(OUTPUT_FOLDER, "esse_kernels_checkpoint.npy")

def filter_and_extract():
    scatter_path = os.path.join(OUTPUT_FOLDER, "phantom_scatters.root")
    spect_path = os.path.join(OUTPUT_FOLDER, "spect.root")
    
    if not (os.path.exists(scatter_path) and os.path.exists(spect_path)):
        return None, None, 0

    with uproot.open(spect_path) as f:
        tree = f["peak208"]
        total_photons = tree.num_entries
        if total_photons == 0:
            return None, None, 0
        det_arrays = tree.arrays(["EventID", "Weight"], library="np")

    df_det = pd.DataFrame(det_arrays).drop_duplicates('EventID').set_index('EventID')
    extracted_points = []
    
    branches_to_read = [
        "EventID", 'PostPosition_X', 'PostPosition_Y', 'PostPosition_Z', 
        "ProcessDefinedStep", "KineticEnergy", 'PostDirection_Z'
    ]
    
    with uproot.open(scatter_path) as f_scat:
        tree_water = f_scat["Hits_Waterbox"]
        for chunk_dict in tree_water.iterate(branches_to_read, step_size="100MB", library="np"):
            valid_event_mask = np.isin(chunk_dict["EventID"], df_det.index.values) & (chunk_dict['PostDirection_Z'] > 0.999)
            if not np.any(valid_event_mask):
                continue

            event_ids = chunk_dict["EventID"][valid_event_mask]
            proc_steps = chunk_dict["ProcessDefinedStep"][valid_event_mask].astype(str) 
            compt_mask = np.char.find(proc_steps, 'compt') != -1
            
            if not np.any(compt_mask):
                continue

            useful_df = pd.DataFrame({
                'EventID': event_ids[compt_mask],
                'PostPosition_X': chunk_dict['PostPosition_X'][valid_event_mask][compt_mask],
                'PostPosition_Y': chunk_dict['PostPosition_Y'][valid_event_mask][compt_mask],
                'PostPosition_Z': chunk_dict['PostPosition_Z'][valid_event_mask][compt_mask],
                'KineticEnergy': chunk_dict['KineticEnergy'][valid_event_mask][compt_mask]
            })

            if useful_df.empty:
                continue
        
            # On ne garde que la DERNIÈRE interaction de diffusion (le point d'émission effectif)
            last_hits = useful_df.groupby('EventID', sort=False).tail(1).set_index('EventID')
            last_hits['Weight'] = df_det['Weight']
            
            # Énergie en keV et évaluation de mu après diffusion
            energy_kev = last_hits['KineticEnergy'] * 1000.0
            last_hits['mu_i'] = get_mu_water(energy_kev)
                
            # 1. Calcul de la distance d'atténuation restante vers le détecteur (cm)
            d_photon = (WATERBOX_EXIT_Z - last_hits['PostPosition_Z']) / 10.0
                
            # 2. Poids ESSE = Désatténuation du trajet de sortie (Slide ESSE)
            last_hits['ESSE_Weight'] = last_hits['Weight'] * np.exp(MU_WATER * d_photon)
            
            # 3. Calcul du numérateur Delta Mu : Poids * (mu_i - mu_water)
            # CORRECTION CRITIQUE : on prend bien la différence de Mu, pas juste mu_i
            last_hits['ESSE_Delta_Mu_Weight'] = (last_hits['mu_i'] - MU_WATER) * last_hits['ESSE_Weight']
                
            # 4. Coordonnées relatives à la source (Nuage 3D centré)
            last_hits['Rel_X'] = last_hits['PostPosition_X'] # Source à 0
            last_hits['Rel_Y'] = last_hits['PostPosition_Y'] # Source à 0
            last_hits['Rel_Z'] = last_hits['PostPosition_Z'] - SOURCE_Z_MM
                
            extracted_points.append(last_hits)

    if not extracted_points:
        return None, None, total_photons

    df_final = pd.concat(extracted_points)
    
    # Limites spatiales pour l'histogramme (en mm)
    limit_xy_mm = (KRNL_SIZE_XY * PIXEL_SIZE * 10) / 2.0
    # Pour Z, on prend une plage suffisamment grande pour capturer le nuage avant/arrière
    limit_z_mm = (KRNL_SIZE_Z * PIXEL_SIZE * 10) / 2.0 
    
    ranges_3d = [[-limit_z_mm, limit_z_mm], [-limit_xy_mm, limit_xy_mm], [-limit_xy_mm, limit_xy_mm]]
    bins_3d = [KRNL_SIZE_Z, KRNL_SIZE_XY, KRNL_SIZE_XY]

    # --- HISTOGRAMMES 3D (Z, Y, X pour matcher PyTorch) ---
    
    # Dénominateur (k0 brut)
    h_k0_sum, _ = np.histogramdd(
        (df_final['Rel_Z'], df_final['Rel_Y'], df_final['Rel_X']),
        bins=bins_3d, range=ranges_3d, weights=df_final['ESSE_Weight']
    ) 

    # Numérateur (pour calculer a_mu)
    h_amu_num_sum, _ = np.histogramdd(
        (df_final['Rel_Z'], df_final['Rel_Y'], df_final['Rel_X']),
        bins=bins_3d, range=ranges_3d, weights=df_final['ESSE_Delta_Mu_Weight']
    )

    return h_k0_sum, h_amu_num_sum, total_photons

# --- INITIALISATION ---
k0_accum = np.zeros((KRNL_SIZE_Z, KRNL_SIZE_XY, KRNL_SIZE_XY))
amu_num_accum = np.zeros((KRNL_SIZE_Z, KRNL_SIZE_XY, KRNL_SIZE_XY))

start_run = 0

if os.path.exists(checkpoint_path):
    print(f">>> Chargement du checkpoint : {checkpoint_path}")
    cp = np.load(checkpoint_path, allow_pickle=True).item()
    k0_accum = cp['k0']
    amu_num_accum = cp['amu_num']
    norm_factor = cp['norm_factor']
    start_run = cp['last_run'] + 1
    print(f"Reprise à partir du Run {start_run + 1}")
else:
    print(">>> Phase de normalisation : Simulation dans l'AIR")
    nb_air_total = 0
    AIR_RUNS = 5

    for a in range(AIR_RUNS):
        env = os.environ.copy()
        env["SOURCE_Z_POS"] = "0"
        subprocess.run([sys.executable, SIM_SCRIPT_AIR], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT_AIR))
        spect_air_path = os.path.join(OUTPUT_FOLDER, "spect.root")
        if os.path.exists(spect_air_path):
            with uproot.open(spect_air_path) as f:
                nb_air_total += f["peak208"].num_entries
            os.remove(spect_air_path)

    # Facteur de normalisation global (ramené au nombre de runs total)
    norm_factor = (nb_air_total / AIR_RUNS) * TOTAL_RUNS
    print(f"Facteur de normalisation calculé : {norm_factor} photons")

# --- GÉNÉRATION DU KERNEL 3D (SOURCE UNIQUE) ---
for r_idx in range(start_run, TOTAL_RUNS):
    print(f"\n=== Lancement RUN {r_idx+1}/{TOTAL_RUNS} ===")
    
    env = os.environ.copy()
    env["SOURCE_Z_POS"] = str(SOURCE_Z_MM)

    try:
        subprocess.run([sys.executable, SIM_SCRIPT], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT))
        
        h_k0, h_amu_num, _ = filter_and_extract()
        
        if h_k0 is not None:
            k0_accum += h_k0
            amu_num_accum += h_amu_num

        # Nettoyage
        for f in ["spect.root", "phantom_scatters.root"]:
            p = os.path.join(OUTPUT_FOLDER, f)
            if os.path.exists(p): os.remove(p)

        # Checkpoint
        if (r_idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_data = {
                'last_run': r_idx,
                'k0': k0_accum,
                'amu_num': amu_num_accum,
                'norm_factor': norm_factor
            }
            temp_path = checkpoint_path + ".tmp"
            np.save(temp_path, checkpoint_data)
            os.replace(temp_path + ".npy", checkpoint_path)
            print(f"[Checkpoint] Run {r_idx+1} sauvegardé.")

    except Exception as e:
        print(f"!!! Erreur critique Run {r_idx}: {e}")

# --- POST-TRAITEMENT MATHÉMATIQUE ESSE ---
print("\n>>> Calcul des 3 composantes ESSE (k0, k1, k2)...")

# 1. Normalisation du kernel de base (Ordre 0)
k0_final = k0_accum / norm_factor

# 2. Calcul du ratio a_mu (sécurisé contre la division par zéro)
a_mu = np.divide(amu_num_accum, k0_accum, out=np.zeros_like(k0_accum), where=k0_accum != 0)

# 3. Création des ordres 1 et 2 selon Taylor
k1_final = k0_final * a_mu
k2_final = k0_final * (a_mu ** 2)

# 4. Assemblage du tenseur 4D final pour PyTorch: shape = [3, Z, Y, X]
esse_tensor_4d = np.stack([k0_final, k1_final, k2_final], axis=0)

# --- SAUVEGARDE ET DIAGNOSTIC ---
np.save(os.path.join(OUTPUT_FOLDER, "esse_kernels_ready_for_pytorch.npy"), esse_tensor_4d)
print(f"\n✅ Tenseur ESSE 4D généré avec succès ! Shape : {esse_tensor_4d.shape}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(k0_final[KRNL_SIZE_Z//2, :, :], cmap='hot')
plt.title("k0 (Base Scatter)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(k1_final[KRNL_SIZE_Z//2, :, :], cmap='magma')
plt.title("k1 (Delta Mu)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(k2_final[KRNL_SIZE_Z//2, :, :], cmap='viridis')
plt.title("k2 (Delta Mu ^2)")
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "diag_esse_3d.png"))