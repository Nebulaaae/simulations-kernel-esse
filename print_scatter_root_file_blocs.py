import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les IDs détectés
with uproot.open("output/spect.root") as detector_file:
    # On utilise 'peak208' pour être plus proche de la méthode Frey
    df_det = detector_file["peak208"].arrays(["EventID", "TrackID"], library="pd")
    detected_ids = df_det.drop_duplicates(['EventID', 'TrackID'])

# Parcourir le fichier Waterbox par Chunks
waterbox_path = "output/phantom_scatters.root:Hits_Waterbox"
final_scatters_list = []

print("Traitement par blocs...")


for chunk in uproot.iterate(waterbox_path, 
                            ["EventID", "TrackID", "PostPosition_X", "PostPosition_Y", "PostPosition_Z", "ProcessDefinedStep"], 
                            step_size="100 MB", library="pd"):
    
    # Correction de type
    chunk['ProcessDefinedStep'] = chunk['ProcessDefinedStep'].astype(str)
    
    # Filtrage Compton
    chunk_scatters = chunk[chunk['ProcessDefinedStep'].str.contains('compt', na=False)]
    
    # Jointure
    useful_chunk = pd.merge(chunk_scatters, detected_ids[['EventID']], on='EventID')
    
    if not useful_chunk.empty:
        print(f"-> {len(useful_chunk)} lignes de scatter trouvées pour ce bloc sur {len(chunk_scatters)} interactions compton.")
        final_scatters_list.append(useful_chunk)


if final_scatters_list:
    df_all_scatters = pd.concat(final_scatters_list)
    last_scatters = df_all_scatters.groupby('EventID').tail(1)
    
    print(f"Extraction réussie : {len(last_scatters)} points de kernels trouvés.")

    # Visualisation
    plt.figure(figsize=(10, 8))
    plt.hist2d(last_scatters['PostPosition_X'], 
               last_scatters['PostPosition_Y'], 
               bins=128, range=[[-150, 150], [-150, 150]], cmap='hot')
    plt.colorbar(label='Nombre de derniers scatters détectés')
    plt.show()

    # 4. Sauvegarde dans un nouveau fichier ROOT
    output_filename = "output/filtered_scatters.root"

    print(f"Sauvegarde en cours dans {output_filename}...")

    # Créer le fichier et écrire l'arbre
    with uproot.recreate(output_filename) as f:
            # On transforme le DataFrame en dictionnaire pour uproot
            # .to_dict('list') permet de passer les colonnes proprement
        f["Filtered_Scatters"] = last_scatters.to_dict('list')

    print("Fichier ROOT généré avec succès !")

else:
    print("Aucun scatter détecté trouvé dans les blocs.")
    
  