import uproot
import numpy as np
import os
import subprocess
import sys
import SimpleITK as sitk
import json

# --- CONFIGURATION ---
IMG_SIZE = 128
PIXEL_SIZE = 0.44 
NB_ANGLES = 5
RUNS_PER_ANGLE = 2
ROR = 25.0             
ANGLES = np.linspace(0, 360, NB_ANGLES, endpoint=False)

OUTPUT_FOLDER = os.path.abspath("./output_spect")
SIM_SCRIPT = os.path.abspath("./simulation2.py")
CHECKPOINT_PATH = os.path.join(OUTPUT_FOLDER, "spect_checkpoint.npy")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_and_separate():
    """Lit les fichiers ROOT et sépare Primaire/Scatter."""
    scatter_path = os.path.join(OUTPUT_FOLDER, "phantom_scatters_gt.root")
    spect_path = os.path.join(OUTPUT_FOLDER, "spect_hits.root")
    
    if not (os.path.exists(scatter_path) and os.path.exists(spect_path)):
        return None, None

    with uproot.open(spect_path) as f:
        tree = f["peak208"]
        df_det = tree.arrays(["EventID", "PostPosition_X", "PostPosition_Y"], library="pd")

    if df_det.empty: return np.zeros((IMG_SIZE, IMG_SIZE)), np.zeros((IMG_SIZE, IMG_SIZE))

    scatter_ids = set()
    with uproot.open(scatter_path) as f:
        tree_scat = f["phantom_scatters_gt"]
        # On itère par blocs
        for chunk in tree_scat.iterate(["EventID", "ProcessDefinedStep"], step_size="50MB", library="pd"):
            # Filtrage : l'événement est détecté ET a subi un 'compt' (Compton)
            mask = (chunk['ProcessDefinedStep'].astype(str).str.contains('compt'))
            relevant_ids = chunk.loc[mask, 'EventID']
            scatter_ids.update(relevant_ids)

    mask_scatter = df_det['EventID'].isin(scatter_ids)
    df_scatter = df_det[mask_scatter]
    df_primary = df_det[~mask_scatter]

    limit = (IMG_SIZE * PIXEL_SIZE) / 2
    range_img = [[-limit, limit], [-limit, limit]]
    
    h_prim, _, _ = np.histogram2d(df_primary['PostPosition_X'], df_primary['PostPosition_Y'], bins=IMG_SIZE, range=range_img)
    h_scat, _, _ = np.histogram2d(df_scatter['PostPosition_X'], df_scatter['PostPosition_Y'], bins=IMG_SIZE, range=range_img)

    return h_prim, h_scat

# --- INITIALISATION ---
volume_primary = np.zeros((NB_ANGLES, IMG_SIZE, IMG_SIZE))
volume_scatter = np.zeros((NB_ANGLES, IMG_SIZE, IMG_SIZE))
start_angle_idx = 0

if os.path.exists(CHECKPOINT_PATH):
    cp = np.load(CHECKPOINT_PATH, allow_pickle=True).item()
    start_angle_idx = cp['next_idx']
    volume_primary = cp['vol_p']
    volume_scatter = cp['vol_s']

# --- BOUCLE PRINCIPALE ---
for i in range(start_angle_idx, NB_ANGLES):
    angle = ANGLES[i]
    h_prim_angle = np.zeros((IMG_SIZE, IMG_SIZE))
    h_scat_angle = np.zeros((IMG_SIZE, IMG_SIZE))
    
    for r in range(RUNS_PER_ANGLE):
        env = os.environ.copy()
        env["SPECT_ANGLE"] = str(angle)
        env["BATCH_ID"] = str(i * RUNS_PER_ANGLE + r) 

        subprocess.run([sys.executable, SIM_SCRIPT], env=env, check=True)
        
        hp, hs = extract_and_separate()
        if hp is not None:
            h_prim_angle += hp
            h_scat_angle += hs
            
        # NETTOYAGE IMMEDIAT
        for f in ["spect_hits.root", "phantom_scatters_gt.root"]:
            p = os.path.join(OUTPUT_FOLDER, f)
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

with open(os.path.join(OUTPUT_FOLDER, "metadata_pytomography.json"), "w") as f:
    json.dump({"num_projections": NB_ANGLES, "pixel_size_cm": PIXEL_SIZE}, f)

print("Terminé.")