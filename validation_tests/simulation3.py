import opengate as gate
import opengate.contrib.spect.ge_discovery_nm670 as spect_ge_nm670
import opengate.contrib.phantoms.nemaiec as nema_p
from opengate.actors.filters import GateFilterBuilder
from scipy.spatial.transform import Rotation as R
import numpy as np
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
sim.g4_verbose = True
sim.visu = False
sim.progress_bar = True
sim.number_of_threads = threads
sim.output_dir = "./nema_final_sim"
sim.random_seed = 12345 + batch_id

mm = gate.g4_units.mm
cm = gate.g4_units.cm
keV = gate.g4_units.keV
sec = gate.g4_units.second
Bq = gate.g4_units.Bq
MBq = 1e6 * gate.g4_units.Bq

# --- Monde ---
sim.world.size = [2 * gate.g4_units.m] * 3
sim.world.material = "G4_AIR"

# --- Fantôme NEMA IEC ---
phantom = nema_p.add_iec_phantom(sim, "nema")
phantom.user_info.translation = [[0, 0, 0]]
rot_flip = R.from_euler('x', 180, degrees=True).as_matrix()
phantom.user_info.rotation = [rot_flip]

# --- Configuration SPECT ---
spect, colli, crystal = spect_ge_nm670.add_spect_head(sim, "spect", "megp")
rad = 40 * cm
pos_x = rad * np.sin(np.radians(current_angle))
pos_z = rad * np.cos(np.radians(current_angle))

spect.user_info.translation = [[pos_x, 0, pos_z]]
rot_matrix = R.from_euler('y', 180 + current_angle, degrees=True).as_matrix()
spect.user_info.rotation = [rot_matrix]

# --- Digitizer ---
hc = sim.add_actor("DigitizerHitsCollectionActor", f"Hits_{crystal.name}")
hc.attached_to = crystal.name
hc.attributes = ["EventID", "PostPosition", "TotalEnergyDeposit", "UnscatteredPrimaryFlag"]

channels = [
    {"name": "scatter3", "min": 176.46 * keV, "max": 191.36 * keV},
    {"name": "peak208", "min": 192.4 * keV, "max": 223.6 * keV},
    {"name": "scatter4", "min": 224.64 * keV, "max": 243.3 * keV},
]

cc = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{crystal.name}")
cc.attached_to = crystal.name
cc.input_digi_collection = hc.name
cc.channels = channels
cc.attributes = ["UnscatteredPrimaryFlag"]

# --- Filtrage ---
F = GateFilterBuilder()
filter_primary = (F.UnscatteredPrimaryFlag == True)

# --- Projections ---
common_params = {
    "attached_to": crystal.name,
    "input_digi_collections": ["peak208"],
    "spacing": [4.4 * mm, 4.4 * mm],
    "size": [128, 128]
}

# Projection TOTAL (tous les photons dans les fenêtres)
proj_total = sim.add_actor("DigitizerProjectionActor", "proj_total")
proj_total.attached_to = crystal.name
# proj_total.input_digi_collections = ["scatter3", "peak208", "scatter4"]
proj_total.input_digi_collections = ["peak208"]
proj_total.spacing = [4.4 * mm, 4.4 * mm]
proj_total.size = [128, 128]
proj_total.output_filename = f"proj_total_angle_{int(current_angle)}.mhd"

# Projection PRIMAIRE (Uniquement non-diffusés)
proj_prim = sim.add_actor("DigitizerProjectionActor", "proj_primary")
proj_prim.attached_to = crystal.name
proj_prim.input_digi_collections = ["peak208"]
proj_prim.spacing = [4.4 * mm, 4.4 * mm]
proj_prim.size = [128, 128]
# proj_prim.attributes = ["UnscatteredPrimaryFlag"]
proj_prim.filter = filter_primary
proj_prim.output_filename = f"proj_primary_angle_{int(current_angle)}.mhd"

# Projection SCATTER (Uniquement diffusés)
proj_scat = sim.add_actor("DigitizerProjectionActor", "proj_scatter")
proj_scat.attached_to = crystal.name
proj_scat.input_digi_collections = ["peak208"] 
proj_scat.spacing = [4.4 * mm, 4.4 * mm]
proj_scat.size = [128, 128]
# proj_scat.attributes = ["UnscatteredPrimaryFlag"]
proj_scat.filter = ~filter_primary
proj_scat.output_filename = f"proj_scatter_angle_{int(current_angle)}.mhd"

# --- Sources ---
# Calcul de l'activité basé sur la concentration pour la sphère de 37mm
total_activity_37mm = 4 * Bq 
radius_ref = 18.5 * mm
vol_ref_ml = (4/3) * np.pi * (radius_ref**3) / (1000 * mm**3) # Volume en mL
activity_concentration = total_activity_37mm / vol_ref_ml # Bq/mL

# Utilisation de la fonction du module contrib pour injecter les sources
activities_bq_ml = [activity_concentration] * 6
nema_p.add_spheres_sources(
    sim, 
    "nema", 
    "src", 
    spheres="all", 
    activity_Bq_mL=activities_bq_ml,
    source_type="GenericSource"
)

# Correction manuelle des paramètres sources pour correspondre à votre setup (gamma 208 keV)
for name, src in sim.source_manager.sources.items():
    if "src_nema_sphere" in name:
        src.particle = "gamma"
        src.energy.type = "mono"
        src.energy.mono = 208 * keV

# --- Physique ---
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
sim.run_timing_intervals = [[0, 100 * sec]]

sim.run()