import opengate as gate
import opengate.contrib.spect.ge_discovery_nm670 as spect_ge_nm670
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import SimpleITK as sitk
import glob


# Chemin vers le CT
ct_dir = os.path.expanduser("~/SPECT/Lu177-NEMA-SymT2/CT")

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(ct_dir)
reader.SetFileNames(dicom_names)
ct_image = reader.Execute()

# Sauvegarde nouveau format
sitk.WriteImage(ct_image, "nema_phantom_ct.mhd")

sim = gate.Simulation()

# Paramètres globaux
sim.g4_verbose = False
sim.visu = True
sim.number_of_threads = 8
sim.output_dir = "./nema_simulation_v1"
sim.check_volumes_overlap = False
sim.progress_bar = True

# Unités
mm = gate.g4_units.mm
cm = gate.g4_units.cm
keV = gate.g4_units.keV
MeV = gate.g4_units.MeV
kBq = 1000 * gate.g4_units.Bq
MBq = 1e6 * gate.g4_units.Bq
sec = gate.g4_units.second

# Monde et Fantôme
sim.world.size = [1 * gate.g4_units.m] * 3
sim.world.material = "G4_AIR"

# Fantôme à partir du CT
phantom = sim.add_volume("Image", "nema_phantom")
phantom.image = "nema_phantom_ct.mhd"
phantom.material = "G4_WATER"
phantom.set_materials_from_voxelisation("/home/chloe/simulations-kernel-esse/validation_tests/HounsfieldUnit_to_Material.json")
phantom.user_info.translation = [[0, 0, 0]]

# Génération map CT
# ct_map = sim.add_actor("AttenuationImageActor", "ct_map")
# ct_map.attached_to = "nema_phantom"
# ct_map.output_filename = "nema_mu_map.mhd"
# ct_map.image_volume = phantom
# ct_map.energy = 208 #* keV

# Configuration SPECT
collimator_type = "megp"
spect, colli, crystal = spect_ge_nm670.add_spect_head(sim, "spect", collimator_type)

# Positionnement
spect.user_info.translation = [[0, 0, 40 * cm]]
rot = R.from_euler('y', 180, degrees=True).as_matrix()
spect.user_info.rotation = [rot]

# Digitizer (Simplifié pour la clarté)
channels = [{"name": "peak208", "min": 192.4 * keV, "max": 223.6 * keV}]

hc = sim.add_actor("DigitizerHitsCollectionActor", f"Hits_{crystal.name}")
hc.attached_to = crystal.name
hc.attributes = ["EventID", "PostPosition", "TotalEnergyDeposit", "PostDirection"]
hc.output_filename = "spect_hits.root"

cc = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{crystal.name}")
cc.attached_to = crystal.name
cc.input_digi_collection = hc.name
cc.channels = channels
cc.output_filename = "spect_signals.root"

proj = sim.add_actor("DigitizerProjectionActor", f"Projection_{crystal.name}")
proj.attached_to = crystal.name
proj.input_digi_collections = ["peak208"]
proj.spacing = [4.4 * mm, 4.4 * mm]
proj.size = [128, 128]
proj.output_filename = "projection_gt.mhd"

# --- EXTRACTION DES SCATTERS DANS LE FANTOME ---
phantom_hits = sim.add_actor("DigitizerHitsCollectionActor", "Hits_Phantom")
phantom_hits.attached_to = "nema_phantom"
phantom_hits.attributes = ["EventID", "PostPosition", "ProcessDefinedStep", "KineticEnergy"]
phantom_hits.output_filename = "phantom_scatters_gt.root"

# Volume pour source (pour le moment j'enlève pour éviter overlap ?)
# hot_sphere_vol = sim.add_volume("Sphere", "hot_sphere_37mm")
# hot_sphere_vol.rmax = 18.5 * mm
# hot_sphere_vol.mother = "world"
# hot_sphere_vol.user_info.translation = [[0, -66, 0]]
# hot_sphere_vol.material = "G4_WATER"

# Source
source_sphere = sim.add_source("GenericSource", "lu177_sphere_37mm")
source_sphere.particle = "gamma"
source_sphere.energy.type = "spectrum_discrete"
source_sphere.energy.spectrum_energies = [208.36 * keV] # Pic Lu177 principal
source_sphere.energy.spectrum_weights = [1.0]

source_sphere.attached_to = "world"
source_sphere.position.translation = [0, -66, 0]
source_sphere.position.type = "sphere"
source_sphere.position.radius = 18.5 * mm
# source_sphere.direction.type = "momentum"
# source_sphere.direction.momentum = [0, 0, 1]

source_sphere.direction.type = "iso"
source_sphere.direction.focus_dir = [0, 0, 1]
source_sphere.direction.focus_theta = [0, 15 * gate.g4_units.deg]

# Activité pour cette sphère
# source_sphere.activity = 10 * MBq / sim.number_of_threads
sim.number_of_threads = 1
source_sphere.activity = 9 * gate.g4_units.Bq

# Physique
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"

# Run
sim.run_timing_intervals = [[0, 10 * sec]]
sim.run()