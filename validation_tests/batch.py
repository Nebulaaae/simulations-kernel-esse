import uproot
import numpy as np
import os
import subprocess
import sys
import SimpleITK as sitk
import json
import argparse

parser = argparse.ArgumentParser(description="Lancement du batch SPECT avec option de restauration.")
parser.add_argument("--restore", action="store_true", help="Reprendre à partir du dernier checkpoint s'il existe.")
args_cmd = parser.parse_args()

# --- CONFIGURATION ---
IMG_SIZE = 128
PIXEL_SIZE = 0.44 
NB_ANGLES = 5
RUNS_PER_ANGLE = 4
ROR = 25.0             
ANGLES = np.linspace(0, 360, NB_ANGLES, endpoint=False)

INPUT_FOLDER = os.path.abspath("./nema_final_sim")
OUTPUT_FOLDER = os.path.abspath("./output_spect")
SIM_SCRIPT = os.path.abspath("./simulation2.py")
CHECKPOINT_PATH = os.path.join(OUTPUT_FOLDER, "spect_checkpoint.npy")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_and_separate():
    """Lit les fichiers ROOT et sépare Primaire/Scatter."""
    # print("script d'extraction")
    scatter_path = os.path.join(INPUT_FOLDER, "phantom_scatters.root")
    spect_path = os.path.join(INPUT_FOLDER, "spect_hits.root")
    
    if not (os.path.exists(scatter_path) and os.path.exists(spect_path)):
        print(f"Fichiers ROOT manquants : {scatter_path} ou {spect_path}")
        return None, None

    with uproot.open(spect_path) as f:
        # print("on ouvre le fichier spect_hits.root")
        # tree = f["peak208"]
        tree = f["Hits_spect_crystal"]
        df_det = tree.arrays(["EventID", "PostPosition_X", "PostPosition_Y"], library="pd")

    if df_det.empty: return np.zeros((IMG_SIZE, IMG_SIZE)), np.zeros((IMG_SIZE, IMG_SIZE))

    scatter_ids = set()
    with uproot.open(scatter_path) as f:
        # print("on ouvre le fichier phantom_scatters.root")
        tree_scat = f["Hits_phantom"]
        # On itère par blocs
        for chunk in tree_scat.iterate(["EventID", "ProcessDefinedStep"], step_size="50MB", library="pd"):
            # Filtrage : l'événement est détecté ET a subi un 'compt' (Compton)
            mask = (chunk['ProcessDefinedStep'].astype(str).str.contains('compt'))
            relevant_ids = chunk.loc[mask, 'EventID']
            scatter_ids.update(relevant_ids)
            # print(relevant_ids)
        
        # print(f"Total d'EventID avec scatter de type 'compt': {len(scatter_ids)}")

    mask_scatter = df_det['EventID'].isin(scatter_ids)
    df_scatter = df_det[mask_scatter]
    df_primary = df_det[~mask_scatter]

    df_det['PostPosition_X'].max()

    limit = (IMG_SIZE * PIXEL_SIZE) / 2
    range_img = [[-limit, limit], [-limit, limit]]
    
    h_prim, _, _ = np.histogram2d(df_primary['PostPosition_X'], df_primary['PostPosition_Y'], bins=IMG_SIZE, range=range_img)
    h_scat, _, _ = np.histogram2d(df_scatter['PostPosition_X'], df_scatter['PostPosition_Y'], bins=IMG_SIZE, range=range_img)

    return h_prim, h_scat

# --- INITIALISATION ---
volume_primary = np.zeros((NB_ANGLES, IMG_SIZE, IMG_SIZE))
volume_scatter = np.zeros((NB_ANGLES, IMG_SIZE, IMG_SIZE))
start_angle_idx = 0

if os.path.exists(CHECKPOINT_PATH) and args_cmd.restore:
    cp = np.load(CHECKPOINT_PATH, allow_pickle=True).item()
    start_angle_idx = cp['next_idx']
    volume_primary = cp['vol_p']
    volume_scatter = cp['vol_s']

# --- BOUCLE PRINCIPALE ---
for i in range(start_angle_idx, NB_ANGLES):
    print(f"Traitement de l'angle {ANGLES[i]:.1f}° (index {i})")
    angle = ANGLES[i]
    h_prim_angle = np.zeros((IMG_SIZE, IMG_SIZE))
    h_scat_angle = np.zeros((IMG_SIZE, IMG_SIZE))
    
    for r in range(RUNS_PER_ANGLE):
        print(f"  Run {r + 1}/{RUNS_PER_ANGLE} pour l'angle {angle:.1f}°")
        env = os.environ.copy()
        env["SPECT_ANGLE"] = str(angle)
        env["BATCH_ID"] = str(i * RUNS_PER_ANGLE + r) 

        subprocess.run([sys.executable, SIM_SCRIPT, str(angle), str(i * RUNS_PER_ANGLE + r)], env=env, check=True)
        
        hp, hs = extract_and_separate()
        if hp is not None:
            h_prim_angle += hp
            h_scat_angle += hs
        
        # print(f"  Run {r + 1} terminé pour l'angle {angle:.1f}°")
            
        # NETTOYAGE IMMEDIAT
        for f in ["spect_hits.root", "phantom_scatters.root"]:
            p = os.path.join(INPUT_FOLDER, f)
            if os.path.exists(p): os.remove(p)

    volume_primary[i, :, :] = h_prim_angle
    volume_scatter[i, :, :] = h_scat_angle
    np.save(CHECKPOINT_PATH, {'next_idx': i + 1, 'vol_p': volume_primary, 'vol_s': volume_scatter})

# --- SAUVEGARDE FINALE ---
def save_mhd(data, name):
    img = sitk.GetImageFromArray(data.astype(np.float32))
    img.SetSpacing([PIXEL_SIZE, PIXEL_SIZE, 1.0])
    sitk.WriteImage(img, os.path.join(OUTPUT_FOLDER, name))

save_mhd(volume_primary, "projections_primary.mhd")
save_mhd(volume_scatter, "projections_scatter.mhd")
save_mhd(volume_primary + volume_scatter, "projections_total.mhd")

# --- Dans la section SAUVEGARDE FINALE ---

metadata = {
    "simulation_params": {
        "num_projections": int(NB_ANGLES),
        "pixel_size_cm": float(PIXEL_SIZE),
        "matrix_size": int(IMG_SIZE),
        "runs_per_angle": int(RUNS_PER_ANGLE)
    },
    "geometry": {
        "angles": ANGLES.tolist(),
        "radii_cm": [float(ROR)] * int(NB_ANGLES),
        "direction": "CCW" 
    },
    "reconstruction_info": {
        "collimator": "MEGP",
        "energy_kev": 208.0,
        "intrinsic_resolution_cm": 0.38
    }
}

with open(os.path.join(OUTPUT_FOLDER, "metadata_pytomography.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("Terminé.")