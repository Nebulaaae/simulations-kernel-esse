import opengate as gate
import uproot
import pandas as pd
import os
import shutil
import subprocess
import sys
import opengate.contrib.spect.ge_discovery_nm670 as spect_ge_nm670

# --- CONFIGURATION DE LA BOUCLE ---
NB_REPETITIONS = 100
OUTPUT_FOLDER = "./output"
FINAL_DATA_FILE = "kernel_accumulated.csv"


def filter_and_extract():
    """Extrait les scatters utiles du run actuel"""
    # 1. Charger les IDs du détecteur (Peak 208)
    with uproot.open(f"{OUTPUT_FOLDER}/spect.root") as f:
        df_det = f["peak208"].arrays(["EventID", "Weight"], library="pd")
        detected_ids = df_det.drop_duplicates(['EventID'])

    # 2. Filtrer le Phantom
    waterbox_path = f"{OUTPUT_FOLDER}/phantom_scatters.root:Hits_Waterbox"
    extracted_points = []
    
    # Lecture par chunk pour la sécurité
    for chunk in uproot.iterate(waterbox_path, ["EventID", "PostPosition_X", "PostPosition_Y", "PostPosition_Z", "ProcessDefinedStep"], library="pd"):
        chunk['ProcessDefinedStep'] = chunk['ProcessDefinedStep'].astype(str)
        mask = (chunk['ProcessDefinedStep'].str.contains('compt')) & (chunk['EventID'].isin(detected_ids['EventID']))
        useful = chunk[mask]
        if not useful.empty:
            merged = useful.merge(detected_ids, on='EventID', how='inner')
            
            if not merged.empty:
                extracted_points.append(merged.groupby('EventID').tail(1))

    if extracted_points:
        return pd.concat(extracted_points)
    return pd.DataFrame()

def cleanup():
    """Supprime les fichiers ROOT volumineux après extraction"""
    files_to_remove = ["spect.root", "phantom_scatters.root", "projection1.mhd", "projection1.raw"]
    for f in files_to_remove:
        path = os.path.join(OUTPUT_FOLDER, f)
        if os.path.exists(path):
            os.remove(path)
    print("Nettoyage des fichiers temporaires effectué.")

# --- BOUCLE PRINCIPALE ---
all_data = []

for i in range(NB_REPETITIONS):
    print(f"\n==========================================")
    print(f" LOG MASTER : Lancement du run {i+1}/{NB_REPETITIONS}")
    print(f"==========================================\n")

    result = subprocess.run([sys.executable, "spect_main1.py"], check=True)
    if result.returncode == 0:
        print(f"\n--- Run {i+1} terminé avec succès ---")
    else:
        print(f"\n--- Erreur lors du run {i+1} ---")
    
    # 2. Filtrer
    try:
        df_batch = filter_and_extract()
        
        if not df_batch.empty:
            # Chemin du fichier final
            final_path = os.path.join(OUTPUT_FOLDER, FINAL_DATA_FILE)
            
            # Vérifier si le fichier existe déjà pour savoir s'il faut écrire l'en-tête
            file_exists = os.path.isfile(final_path)
            
            # Sauvegarde immédiate
            df_batch.to_csv(final_path, 
                            mode='a', 
                            index=False, 
                            header=not file_exists,
                            encoding='utf-8')
            
            print(f"Extraction : {len(df_batch)} points ajoutés au CSV.")
    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")        
    # 3. Nettoyer
    cleanup()
