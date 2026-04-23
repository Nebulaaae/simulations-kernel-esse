import numpy as np
import os
import subprocess
import sys
import SimpleITK as sitk
import json
import argparse

parser = argparse.ArgumentParser(description="Lancement du batch SPECT avec filtres natifs.")
parser.add_argument("--restore", action="store_true", help="Reprendre à partir du dernier checkpoint.")
args_cmd = parser.parse_args()

# --- CONFIGURATION ---
IMG_SIZE = 128
PIXEL_SIZE = 0.44 
NB_ANGLES = 64
ROR = 40.0              
ANGLES = np.linspace(0, 360, NB_ANGLES, endpoint=False)

INPUT_FOLDER = os.path.abspath("./nema_final_sim")
OUTPUT_FOLDER = os.path.abspath("./output_spect")
SIM_SCRIPT = os.path.abspath("./simulation3.py")
CHECKPOINT_PATH = os.path.join(OUTPUT_FOLDER, "batch_checkpoint.npy")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- INITIALISATION DES VOLUMES ---
# PyTomography attend (Angles, Y, X)
vol_p = np.zeros((NB_ANGLES, IMG_SIZE, IMG_SIZE))
vol_s = np.zeros((NB_ANGLES, IMG_SIZE, IMG_SIZE))
vol_t = np.zeros((NB_ANGLES, IMG_SIZE, IMG_SIZE))
vol_t_s3 = np.zeros((NB_ANGLES, IMG_SIZE, IMG_SIZE))
vol_t_s4 = np.zeros((NB_ANGLES, IMG_SIZE, IMG_SIZE))
start_idx = 0

if args_cmd.restore and os.path.exists(CHECKPOINT_PATH):
    cp = np.load(CHECKPOINT_PATH, allow_pickle=True).item()
    start_idx = cp['next_idx']
    vol_p, vol_s, vol_t, vol_t_s3, vol_t_s4 = cp['p'], cp['s'], cp['t'], cp['t_s3'], cp['t_s4']

# --- BOUCLE PRINCIPALE ---
for i in range(start_idx, NB_ANGLES):
    angle = ANGLES[i]

    if os.path.exists(INPUT_FOLDER):
        for f in os.listdir(INPUT_FOLDER):
            if f.endswith(".mhd") or f.endswith(".raw") or f.endswith(".root"):
                os.remove(os.path.join(INPUT_FOLDER, f))

    print(f">>> Simulation Angle {angle:.1f}° ({i+1}/{NB_ANGLES})")
    
    # Exécution de la simulation OpenGate pour cet angle
    # On passe l'angle et un batch_id pour la seed
    subprocess.run([sys.executable, SIM_SCRIPT, str(angle), str(i), "0", "8"], check=True)

    # Récupération des fichiers générés par les filtres natifs
    base_names = {
        'p': f"proj_primary_angle_{int(angle)}_counts.mhd",
        's': f"proj_scatter_angle_{int(angle)}_counts.mhd",
        't': f"proj_total_angle_{int(angle)}_counts.mhd"
    }

    try:
        # Lecture des images
        img_t_raw = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(INPUT_FOLDER, base_names['t'])))
        img_p_raw = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(INPUT_FOLDER, base_names['p'])))
        img_s_raw = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(INPUT_FOLDER, base_names['s'])))
        # img_t_s3_raw = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(INPUT_FOLDER, f"proj_scatter3_angle_{int(angle)}.mhd")))
        # img_t_s4_raw = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(INPUT_FOLDER, f"proj_scatter4_angle_{int(angle)}.mhd")))
        
        # Squeeze pour s'assurer d'être en 2D (128, 128)
        # t_2d = np.squeeze(img_t_raw)


        p_2d = np.squeeze(img_p_raw)
        s_2d = np.squeeze(img_s_raw)
        #pour t, il est de dimension (3, 128, 128) à cause des 3 projections (scatter3, peak208, scatter4), les séparer depuis l'image img_t_raw, pour générer trois imges 2D : t_2d_s3, t_2d_p, t_2d_s4
        t_2d_s3 = np.squeeze(img_t_raw[0, :, :])
        t_2d_p = np.squeeze(img_t_raw[1, :, :])
        t_2d_s4 = np.squeeze(img_t_raw[2, :, :])

        

        # --- CALCULS DE DIAGNOSTIC ---
        # sum_t = np.sum(t_2d)
        # sum_p = np.sum(p_2d)
        # sum_s = np.sum(s_2d)
        
        # print(f"\n[DIAGNOSTIC ANGLE {angle:.1f}°]")
        # print(f"  -> Total Comptes (Total) : {sum_t:.0f}")
        # print(f"  -> Total Primary Comptes : {sum_p:.0f}")
        # print(f"  -> Total Scatter Comptes : {sum_s:.0f}")

        # if sum_p == sum_s and sum_p > 0:
        #     print("  ⚠️ ALERT: Primary et Scatter sont IDENTIQUES au photon près.")
        #     print("     Le filtre ne fonctionne pas.")
        # elif sum_p > 0:
        #     ratio = (sum_s / sum_t) * 100
        #     print(f"  ✅ Ratio Scatter/Total : {ratio:.2f}%")
        
        # # Vérification de la rotation
        # # On regarde où se trouve le centre de masse pour vérifier que l'objet bouge
        # coords = np.argwhere(p_2d > (p_2d.max() * 0.5))
        # if len(coords) > 0:
        #     center = coords.mean(axis=0)
        #     print(f"  -> Centre de masse (Y, X) : ({center[0]:.1f}, {center[1]:.1f})")

        vol_p[i], vol_s[i], vol_t[i], vol_t_s3[i], vol_t_s4[i] = p_2d, s_2d, t_2d_p, t_2d_s3, t_2d_s4

    except Exception as e:
        print(f"  ❌ Erreur lecture : {e}")

    # Nettoyage des fichiers temporaires pour cet angle
    # for f in base_names.values():
    #     p = os.path.join(INPUT_FOLDER, f)
    #     raw = p.replace(".mhd", ".raw")
    #     if os.path.exists(p): os.remove(p)
    #     if os.path.exists(raw): os.remove(raw)
    
    # Checkpoint
    np.save(CHECKPOINT_PATH, {'next_idx': i + 1, 'p': vol_p, 's': vol_s, 't': vol_t, 't_s3': vol_t_s3, 't_s4': vol_t_s4})

# --- ATTÉNUATION ET MÉTADONNÉES ---
print("Génération de la carte d'atténuation...")
subprocess.run([sys.executable, "simulation_attenuation_map.py"], check=True)

def save_final_mhd(data, name):
    img = sitk.GetImageFromArray(data.astype(np.float32))
    img.SetSpacing([PIXEL_SIZE, PIXEL_SIZE, 1.0])
    origin = -(IMG_SIZE * PIXEL_SIZE) / 2.0
    img.SetOrigin([origin, origin, 0.0])
    sitk.WriteImage(img, os.path.join(OUTPUT_FOLDER, name))

save_final_mhd(vol_p, "projections_primary.mhd")
save_final_mhd(vol_s, "projections_scatter.mhd")
save_final_mhd(vol_t, "projections_total.mhd")
save_final_mhd(vol_t_s3, "projections_scatter3.mhd")
save_final_mhd(vol_t_s4, "projections_scatter4.mhd")

# Extraction auto des slices Z de la mu-map
path_mu = os.path.join(OUTPUT_FOLDER, "../nema_maps/nema_mu_map_208keV.mhd")
mu_img = sitk.ReadImage(path_mu)
num_z = mu_img.GetSize()[2]

metadata = {
    "simulation_params": {
        "num_projections": int(NB_ANGLES),
        "pixel_size_cm": float(PIXEL_SIZE),
        "matrix_size": int(IMG_SIZE), 
        "num_slices_z": int(num_z), 
        "voxel_size_cm": float(mu_img.GetSpacing()[0] / 10.0)   
    },
    "geometry": {
        "angles": ANGLES.tolist(),
        "radii_cm": [float(ROR)] * int(NB_ANGLES),
        "direction": "CCW"
    },
    "reconstruction_info": {
        "collimator": "Medium Energy", 
        "energy_kev": 208.0,
        "intrinsic_resolution_cm": 0.38,
        "collimator_slope": 0.054,
        "collimator_intercept": 0.13
    }
}

with open(os.path.join(OUTPUT_FOLDER, "metadata_pytomography.json"), "w") as f:
    json.dump(metadata, f, indent=4)