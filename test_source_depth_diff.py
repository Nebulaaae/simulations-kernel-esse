import subprocess
import os
import sys
import numpy as np
import uproot

# --- CONFIG ---
SIM_SCRIPT = "./spect_main2.py"
OUTPUT_FOLDER = "./output"
ITERATIONS = 50
Z_POSITIONS = [140.0, -140.0]

def get_photon_count():
    """Extrait le nombre d'entrées dans l'arbre peak208 du fichier ROOT"""
    root_path = os.path.join(OUTPUT_FOLDER, "spect.root")
    if not os.path.exists(root_path):
        return 0
    try:
        with uproot.open(root_path) as f:
            return f["peak208"].num_entries
    except Exception as e:
        print(f"Erreur lecture ROOT: {e}")
        return 0

results = {pos: [] for pos in Z_POSITIONS}

# --- TEST ---
for i in range(ITERATIONS):
    print(f"\n=== Itération {i+1}/{ITERATIONS} ===")
    
    for z in Z_POSITIONS:
        print(f"Simulation Z = {z} mm...", end=" ", flush=True)
        
        env = os.environ.copy()
        env["SOURCE_Z_POS"] = str(z)
        
        try:
            subprocess.run([sys.executable, SIM_SCRIPT], env=env, check=True, cwd=os.path.dirname(SIM_SCRIPT))
            
            count = get_photon_count()
            results[z].append(count)
            print(f"Terminé. Photons: {count}")
            
            # Nettoyage pour le prochain run
            if os.path.exists(os.path.join(OUTPUT_FOLDER, "spect.root")):
                os.remove(os.path.join(OUTPUT_FOLDER, "spect.root"))
                
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de la simulation: {e.stderr}")

# --- RÉSULTATS ---
print("Résultats des simulations :")

for z in Z_POSITIONS:
    data = np.array(results[z])
    print(f"\nPosition Z = {z} mm :")
    print(f"  - Moyenne : {np.mean(data):.2f}")
    print(f"  - Écart-type : {np.std(data):.2f}")
    print(f"  - Min/Max : {np.min(data)} / {np.max(data)}")
    print(f"  - Liste brute : {results[z]}")