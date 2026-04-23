import opengate as gate
import opengate.contrib.spect.ge_discovery_nm670 as spect_ge_nm670
import opengate.contrib.phantoms.nemaiec as nema_p
from opengate.actors.filters import GateFilterBuilder  # Import du builder
from scipy.spatial.transform import Rotation as R
import numpy as np
import SimpleITK as sitk
import os
import sys

# --- Arguments ---
try:
    current_angle = float(sys.argv[1])
    batch_id = int(sys.argv[2])
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
except (IndexError, ValueError):
    current_angle = 0.0
    batch_id = 0
    threads = 8

sim = gate.Simulation()

# --- Paramètres globaux ---
sim.g4_verbose = False
sim.visu = False
sim.progress_bar = True
sim.number_of_threads = threads
sim.output_dir = "./nema_final_sim"
sim.random_seed = 12345 + batch_id
sim.check_volumes_overlap = True

mm = gate.g4_units.mm
cm = gate.g4_units.cm
keV = gate.g4_units.keV
sec = gate.g4_units.second
MBq = 1e6 * gate.g4_units.Bq

# --- Monde ---
sim.world.size = [2 * gate.g4_units.m] * 3
sim.world.material = "G4_AIR"

# --- Fantôme NEMA IEC ---
phantom = nema_p.add_iec_phantom(sim, "nema")
phantom.user_info.translation = [[0, 0, 0]]
# rot_flip = R.from_euler('x', 180, degrees=True).as_matrix()
# phantom.user_info.rotation = [rot_flip]

# --- Configuration SPECT ---
spect, colli, crystal = spect_ge_nm670.add_spect_head(sim, "spect", "megp")
rad = 40 * cm

pos_x = rad * np.sin(np.radians(current_angle))
pos_y = rad * np.cos(np.radians(current_angle))
spect.user_info.translation = [[-pos_x, pos_y, 0]]

base_tilt = R.from_euler('x', 90, degrees=True)
orbit_rot = R.from_euler('z', current_angle, degrees=True)
rot_matrix = (orbit_rot * base_tilt).as_matrix()

spect.user_info.rotation = [rot_matrix]

# Rotation de la tête : 
# 1. 'y' est l'axe de rotation.
# 2. On ajoute 180 pour que la face du détecteur regarde vers (0,0,0).
# 3. 'current_angle' doit correspondre au sens de translation.


# --- Digitizer (Hits & Energy Windows) ---
F = GateFilterBuilder()

hc_tot = sim.add_actor("DigitizerHitsCollectionActor", f"Hits_{crystal.name}_tot")
hc_tot.attached_to = crystal.name
hc_tot.attributes = ["EventID", "PostPosition", "TotalEnergyDeposit", "UnscatteredPrimaryFlag"]

hc_prim = sim.add_actor("DigitizerHitsCollectionActor", f"Hits_{crystal.name}_prim")
hc_prim.attached_to = crystal.name
hc_prim.filter = F.UnscatteredPrimaryFlag
hc_prim.attributes = ["EventID", "PostPosition", "TotalEnergyDeposit", "UnscatteredPrimaryFlag"]

hc_scat = sim.add_actor("DigitizerHitsCollectionActor", f"Hits_{crystal.name}_scat")
hc_scat.attached_to = crystal.name
hc_scat.filter = ~F.UnscatteredPrimaryFlag
hc_scat.attributes = ["EventID", "PostPosition", "TotalEnergyDeposit", "UnscatteredPrimaryFlag"]

# Fenêtres d'énergie TEW pour le pic 208 keV
channels = [
    {"name": "scatter3", "min": 176.46 * keV, "max": 191.36 * keV},
    {"name": "peak208", "min": 192.4 * keV, "max": 223.6 * keV},
    {"name": "scatter4", "min": 224.64 * keV, "max": 243.3 * keV},
]

cc_tot = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{crystal.name}_tot")
cc_tot.attached_to = crystal.name
cc_tot.input_digi_collection = hc_tot.name
# cc_tot.channels = [{"name": "peak_tot", "min": 192.4 * keV, "max": 223.6 * keV}]
cc_tot.channels = channels
cc_tot.attributes = ["UnscatteredPrimaryFlag"]
cc_tot.output_filename = "spect_hits_tot.root"

# cc_tot_scatter3 = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{crystal.name}_scatter3")
# cc_tot_scatter3.attached_to = crystal.name
# cc_tot_scatter3.input_digi_collection = hc_tot.name
# cc_tot_scatter3.channels = [{"name": "scatter3", "min": 176.46 * keV, "max": 191.36 * keV}]
# cc_tot_scatter3.attributes = ["UnscatteredPrimaryFlag"]
# cc_tot_scatter3.output_filename = "spect_hits_scatter3.root"

# cc_tot_scatter4 = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{crystal.name}_scatter4")
# cc_tot_scatter4.attached_to = crystal.name
# cc_tot_scatter4.input_digi_collection = hc_tot.name
# cc_tot_scatter4.channels = [{"name": "scatter4", "min": 224.64 * keV, "max": 243.3 * keV}]
# cc_tot_scatter4.attributes = ["UnscatteredPrimaryFlag"]
# cc_tot_scatter4.output_filename = "spect_hits_scatter4.root"

cc_prim = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{crystal.name}_prim")
cc_prim.attached_to = crystal.name
cc_prim.input_digi_collection = hc_prim.name
cc_prim.channels = [{"name": "peak_prim", "min": 192.4 * keV, "max": 223.6 * keV}]
cc_prim.attributes = ["UnscatteredPrimaryFlag"]
cc_prim.output_filename = "spect_hits_prim.root"

cc_scat = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{crystal.name}_scat")
cc_scat.attached_to = crystal.name
cc_scat.input_digi_collection = hc_scat.name
cc_scat.channels = [{"name": "peak_scat", "min": 192.4 * keV, "max": 223.6 * keV}]
cc_scat.attributes = ["UnscatteredPrimaryFlag"]
cc_scat.output_filename = "spect_hits_scat.root"

# --- FILTRAGE ---

# 1. Filtre pour les photons primaires
filter_primary = (F.UnscatteredPrimaryFlag == True)


# 2. Filtre pour les photons diffusés
filter_scatter = ~F.UnscatteredPrimaryFlag


# --- Projections séparées ---

# Projection TOTAL (tous les photons dans les fenêtres)
proj_tot = sim.add_actor("DigitizerProjectionActor", "proj_tot")
proj_tot.attached_to = crystal.name
proj_tot.input_digi_collections = ["scatter3", "peak208", "scatter4"]
# proj_tot.input_digi_collections = ["peak_tot"]
proj_tot.spacing = [4.4 * mm, 4.4 * mm]
proj_tot.size = [128, 128]
proj_tot.output_filename = f"proj_total_angle_{int(current_angle)}.mhd"

# proj_tot_scatter3 = sim.add_actor("DigitizerProjectionActor", "proj_tot_scatter3")
# proj_tot_scatter3.attached_to = crystal.name
# proj_tot_scatter3.input_digi_collections = ["scatter3"]
# proj_tot_scatter3.spacing = [4.4 * mm, 4.4 * mm]
# proj_tot_scatter3.size = [128, 128]
# proj_tot_scatter3.output_filename = f"proj_scatter3_angle_{int(current_angle)}.mhd"

# proj_tot_scatter4 = sim.add_actor("DigitizerProjectionActor", "proj_tot_scatter4")
# proj_tot_scatter4.attached_to = crystal.name
# proj_tot_scatter4.input_digi_collections = ["scatter4"]
# proj_tot_scatter4.spacing = [4.4 * mm, 4.4 * mm]
# proj_tot_scatter4.size = [128, 128]
# proj_tot_scatter4.output_filename = f"proj_scatter4_angle_{int(current_angle)}.mhd"

# B. Projection PRIMAIRE (Uniquement non-diffusés)
proj_prim = sim.add_actor("DigitizerProjectionActor", "proj_primary")
proj_prim.attached_to = crystal.name
proj_prim.input_digi_collections = ["peak_prim"]
proj_prim.spacing = [4.4 * mm, 4.4 * mm]
proj_prim.size = [128, 128]
proj_prim.output_filename = f"proj_primary_angle_{int(current_angle)}.mhd"

# C. Projection SCATTER (Uniquement diffusés)
proj_scat = sim.add_actor("DigitizerProjectionActor", "proj_scatter")
proj_scat.attached_to = crystal.name
proj_scat.input_digi_collections = ["peak_scat"] 
proj_scat.spacing = [4.4 * mm, 4.4 * mm]
proj_scat.size = [128, 128]
proj_scat.output_filename = f"proj_scatter_angle_{int(current_angle)}.mhd"

# --- Sources ---
total_activity_37mm = 2 * MBq / sim.number_of_threads
radius_ref = 18.5 * mm
vol_ref = (4/3) * np.pi * (radius_ref**3)
concentration = total_activity_37mm / vol_ref
diameters = [10, 13, 17, 22, 28, 37]
diameters = [37]

for d in diameters:
    r = (d / 2) * mm
    sphere_activity = concentration * (4/3) * np.pi * (r**3)
    src = sim.add_source("GenericSource", f"source_sphere_{d}mm")
    src.particle = "gamma"
    src.energy.type = "mono"
    src.energy.mono = 208 * keV
    src.attached_to = f"nema_sphere_{d}mm"
    src.position.type = "sphere"
    src.position.radius = r
    # src.position.fill = True
    src.direction.type = "iso"
    src.activity = sphere_activity

# --- Physique & Run ---
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
sim.run_timing_intervals = [[0, 100 * sec]]
sim.run()

# diagnostic
# import uproot
# for file in ["spect_hits_tot.root", "spect_hits_prim.root", "spect_hits_scat.root"]:
#     print(f"\n--- Contenu de {file} ---")
#     with uproot.open(os.path.join(sim.output_dir, file)) as f:
#         tree = f[f.keys()[0]]  # Récupère le premier arbre (ex: "peak_tot")
#         event_ids = tree["EventID"].array()
#         energy_deposits = tree["TotalEnergyDeposit"].array()
#         primary_flags = tree["UnscatteredPrimaryFlag"].array()
        
#         print(f"Nombre total d'événements enregistrés : {len(event_ids)}")
#         for i in range(min(5, len(event_ids))):  # Affiche les 5 premiers événements
#             print(f"EventID: {event_ids[i]}, TotalEnergyDeposit: {energy_deposits[i]}, UnscatteredPrimaryFlag: {primary_flags[i]}")
