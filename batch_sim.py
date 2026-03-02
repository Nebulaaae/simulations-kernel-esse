import opengate as gate
import uproot
import pandas as pd
import os
import shutil
import subprocess
import sys
import opengate.contrib.spect.ge_discovery_nm670 as spect_ge_nm670

# --- CONFIGURATION DE LA BOUCLE ---
NB_REPETITIONS = 5  # Nombre de fois que la simulation redémarre
OUTPUT_FOLDER = "./output"
FINAL_DATA_FILE = "kernel_accumulated.csv" # On accumule dans un CSV ou un ROOT à la fin

def run_single_sim(activity=1, time_sec=10, nb_threads=1):
    # create the simulation
    sim = gate.Simulation()

    # main options
    sim.g4_verbose = False
    sim.visu = False
    # sim.visu_type = "vrml"
    sim.visu_type = "qt"
    sim.number_of_threads = 6
    sim.random_seed = "auto"
    sim.progress_bar = True
    sim.output_dir = "./output"

    # units
    m = gate.g4_units.m
    sec = gate.g4_units.second
    days = 3600 * 24 * sec
    cm = gate.g4_units.cm
    mm = gate.g4_units.mm
    nm = gate.g4_units.nm
    MeV = gate.g4_units.MeV
    keV = gate.g4_units.keV
    Bq = gate.g4_units.Bq
    kBq = 1000 * Bq
    MBq = 1000 * kBq

    # world size
    sim.world.size = [2 * m, 2 * m, 2 * m]
    sim.world.material = "G4_AIR"

    # waterbox
    wb = sim.add_volume("Box", "waterbox")
    wb.size = [60 * cm, 60 * cm, 30 * cm]
    wb.material = "G4_WATER"
    wb.color = [0, 0, 1, 1]  # blue

    # spect head (debug mode = only a small part of the collimator is simulation,
    # for visu mode)
    # - False: no collimator
    # - lehr: holes length 35 mm, diam 1.5 mm, septal thickness : 0.2 mm
    # - megp: holes length 58 mm, diam 3 mm,   septal thickness : 1.05 mm
    # - hegp: holes length 66 mm, diam 4 mm,   septal thickness : 1.8 mm
    collimator_type = "megp"
    spect, colli, crystal = spect_ge_nm670.add_spect_head(
        sim, "spect", collimator_type, debug=(sim.visu and sim.visu_type != "qt")
    )
    spect_ge_nm670.rotate_gantry(spect, 35*cm, 0)

    # spect digitizer channels
    channels = [
        {"name": f"spectrum", "min": 3 * keV, "max": 515 * keV},
        {"name": f"scatter1", "min": 96 * keV, "max": 104 * keV},
        {"name": f"peak113", "min": 104.52 * keV, "max": 121.48 * keV},
        {"name": f"scatter2", "min": 122.48 * keV, "max": 133.12 * keV},
        {"name": f"scatter3", "min": 176.46 * keV, "max": 191.36 * keV},
        {"name": f"peak208", "min": 192.4 * keV, "max": 223.6 * keV},
        {"name": f"scatter4", "min": 224.64 * keV, "max": 243.3 * keV},
    ]

    # spect digitizer : Hits + Adder + EneWin + Projection
    # Hits
    hc = sim.add_actor("DigitizerHitsCollectionActor", f"Hits_{crystal.name}")
    hc.attached_to = crystal.name
    hc.output_filename = "spect.root"
    hc.attributes = [
        "EventID",
        "TrackID",
        "PostPosition",
        "TotalEnergyDeposit",
        "PreStepUniqueVolumeID",
        "GlobalTime",
        "LocalTime",
        "StepLength",
        "TrackLength",
    ]
    # list of attributes :https://opengate-python.readthedocs.io/en/latest/user_guide.html#actors-and-filters

    # Singles
    sc = sim.add_actor("DigitizerAdderActor", f"Singles_{crystal.name}")
    sc.attached_to = hc.attached_to
    sc.input_digi_collection = hc.name
    sc.policy = "EnergyWinnerPosition"
    sc.output_filename = hc.output_filename

    # energy windows
    cc = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{crystal.name}")
    cc.attached_to = sc.attached_to
    cc.input_digi_collection = sc.name
    cc.channels = channels
    cc.output_filename = hc.output_filename

    # projection image
    proj = sim.add_actor("DigitizerProjectionActor", f"Projection_{crystal.name}")
    proj.attached_to = cc.attached_to
    proj.input_digi_collections = [x["name"] for x in cc.channels]
    proj.spacing = [5 * mm, 5 * mm]
    proj.size = [128, 128]
    proj.output_filename = "projection1.mhd"

    # Acteur pour récupérer les diffusions compton dans le fantôme
    phantom_hits = sim.add_actor("DigitizerHitsCollectionActor", "Hits_Waterbox")
    phantom_hits.attached_to = "waterbox"
    phantom_hits.output_filename = "phantom_scatters.root"

    f = sim.add_filter("ParticleFilter", "gamma_filter")
    f.particle = "gamma"
    phantom_hits.filters.append(f)

    phantom_hits.attributes = [
        "EventID",
        "PostPosition",
        "TrackID",
        "ProcessDefinedStep"
    ]

    # Lu177 source (only the gammas)
    source = sim.add_source("GenericSource", "lu177_gammas")
    source.particle = "gamma"
    source.attached_to = f"waterbox"
    source.energy.type = "spectrum_discrete"
    source.energy.spectrum_weights = [
        0.001726,
        0.0620,
        0.000470,
        0.1038,
        0.002012,
        0.00216,
    ]
    source.energy.spectrum_energies = [
        071.6418 * keV,
        112.9498 * keV,  # 6.2 %
        136.7245 * keV,
        208.3662 * keV,  # 10.38 %
        249.6742 * keV,
        321.3159 * keV,
    ]
    # source.position.type = "sphere"
    source.position.type = "point"
    # source.position.radius = 20 * mm
    source.position.translation = [0, 0, 0 * mm]
    """
    With "iso", the gammas are emitted isotropically, so most of them will not
    been detected. In order to get more signal, you can use "momentum", meaning 
    that the gamma will be emitted with a single direction towards -z axis. Of course, 
    this is not realistic.
    """
    source.direction.type = "iso"
    source.direction.focus_dir = [0, 0, -1]
    source.direction.focus_theta = [0, 90 * gate.g4_units.deg]
    # source.direction.type = "momentum"
    # source.direction.momentum = [0, 1, 0]
    if sim.visu:
        sim.number_of_threads = 4
        source.activity = 100 * Bq
    else:
        sim.number_of_threads = nb_threads
        source.activity = (activity * MBq) / sim.number_of_threads 


    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.track_types_flag = True
    stats.output_filename = "stats1.txt"

    # phys
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)

    # ---------------------------------------------------------------------
    # start simulation
    # sim.running_verbose_level = gate.EVENT
    sim.run_timing_intervals = [[0, time_sec * sec]]


    sim.run()
    return sim

def filter_and_extract():
    """Extrait les scatters utiles du run actuel"""
    # 1. Charger les IDs du détecteur (Peak 208)
    with uproot.open(f"{OUTPUT_FOLDER}/spect.root") as f:
        df_det = f["peak208"].arrays(["EventID"], library="pd")
        detected_ids = df_det.drop_duplicates(['EventID'])

    # 2. Filtrer le Phantom
    waterbox_path = f"{OUTPUT_FOLDER}/phantom_scatters.root:Hits_Waterbox"
    extracted_points = []
    
    # Lecture par chunk pour la sécurité
    for chunk in uproot.iterate(waterbox_path, ["EventID", "PostPosition_X", "PostPosition_Y", "ProcessDefinedStep"], library="pd"):
        chunk['ProcessDefinedStep'] = chunk['ProcessDefinedStep'].astype(str)
        # Filtrage : Uniquement Compton + présent dans le détecteur
        mask = (chunk['ProcessDefinedStep'].str.contains('compt')) & (chunk['EventID'].isin(detected_ids['EventID']))
        useful = chunk[mask]
        if not useful.empty:
            # On ne garde que le dernier scatter par EventID
            extracted_points.append(useful.groupby('EventID').tail(1))

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
            all_data.append(df_batch)
            print(f"Extraction : {len(df_batch)} points ajoutés.")
    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")
    
    # 3. Nettoyer
    cleanup()

# --- SAUVEGARDE FINALE ---
if all_data:
    final_df = pd.concat(all_data).reset_index(drop=True)
    final_df.to_csv(f"{OUTPUT_FOLDER}/{FINAL_DATA_FILE}", index=False)
    print(f"\nTERMINÉ. Total accumulé : {len(final_df)} points de kernel sauvegardés.")
else:
    print("\nAucune donnée n'a pu être accumulée.")