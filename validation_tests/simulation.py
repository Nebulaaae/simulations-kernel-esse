import opengate as gate
import opengate.contrib.spect.ge_discovery_nm670 as spect_ge_nm670
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import SimpleITK as sitk
import glob


# Chemin vers le CT
ct_dir = os.path.expanduser("C:/Users/chloe/Bureau/ICO/SPECT/Lu177-NEMA-SymT2/CT")

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(ct_dir)
reader.SetFileNames(dicom_names)
ct_image = reader.Execute()

print(f"Dimensions CT : {ct_image.GetSize()}")
print(f"Spacing CT : {ct_image.GetSpacing()}")
if np.max(sitk.GetArrayFromImage(ct_image)) <= 0:
    print("ATTENTION : L'image CT est vide ou ne contient que des valeurs <= 0")

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
phantom = sim.add_volume("ImageVolume", "nema_phantom")
phantom.image = "nema_phantom_ct.mhd"
# phantom.material = "G4_WATER"
# Configuration des matériaux avec attributs de couleur [R, G, B, Alpha]
# Alpha = 0 (invisible), Alpha = 1 (opaque)
# phantom.voxel_materials = [
#     [-1025, -900, "G4_AIR", [0, 1, 0, 0]],       # Air : Totalement transparent
#     [-900, 100, "G4_WATER", [1, 0, 0, 0.5]],    # Eau : Rouge semi-transparent
#     [100, 3000, "G4_BONE_COMPACT_ICRU", [1, 1, 1, 1]] # Os : Blanc opaque
# ]

phantom.voxel_materials = [
    [-1025, 90, "G4_AIR", [0, 1, 0, 0]],       # Air : Totalement transparent
    [90, 100, "G4_AIR", [1, 0, 1, 0.5]],    # Eau : Rouge semi-transparent
    [100, 3000, "G4_BONE_COMPACT_ICRU", [1, 1, 1, 1]] # Os : Blanc opaque
]

# Désactivez la couleur globale qui "écrase" les couleurs par voxel
# phantom.user_info.color = [0.5, 0.5, 0.8, 0.3] # À SUPPRIMER
phantom.user_info.translation = [[0, 0, 0]]
phantom.user_info.origin = "center"

ct_array = sitk.GetArrayFromImage(ct_image)
print(f"\n[DIAGNOSTIC CT]")
print(f"Dimensions : {ct_image.GetSize()}")
print(f"Type de pixel : {ct_image.GetPixelIDTypeAsString()}")
print(f"Valeur Min : {np.min(ct_array)}")
print(f"Valeur Max : {np.max(ct_array)}")
print(f"Valeur Moyenne : {np.mean(ct_array)}")


if sim.visu:
    phantom.user_info.vis_attribute = "Wireframe" 
    phantom.user_info.color = [1, 0, 0, 1]

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

source_sphere.attached_to = "nema_phantom"
source_sphere.position.translation = [0, -66 * mm, 0]
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