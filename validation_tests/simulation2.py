import opengate as gate
import opengate.contrib.spect.ge_discovery_nm670 as spect_ge_nm670
import opengate.contrib.phantoms.nemaiec as nema_p
from scipy.spatial.transform import Rotation as R
import numpy as np
import opengate.contrib as contrib
import SimpleITK as sitk
import os
import pkgutil
import sys

# # Liste les sous-packages de contrib
# print("--- Sous-packages de contrib ---")
# for loader, name, ispkg in pkgutil.iter_modules(contrib.__path__):
#     print(f"- {name}")

# # Liste les fichiers dans le dossier phantoms
# phantom_path = os.path.join(contrib.__path__[0], 'phantoms')
# if os.path.exists(phantom_path):
#     print(f"\n--- Contenu de {phantom_path} ---")
#     print(os.listdir(phantom_path))
# else:
#     print(f"\nLe dossier {phantom_path} est introuvable.")

# print([f for f in dir(nema_p) if 'add' in f or 'nema' in f.lower()])
current_angle = float(sys.argv[1]) if len(sys.argv) > 1 else 0
batch_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

support_size = [128, 128, 128]
support_spacing = [3.0, 3.0, 3.0] # mm
air_image = sitk.Image(support_size, sitk.sitkFloat32)
air_image.SetSpacing(support_spacing)
air_image.SetOrigin([-(s*sp)/2 for s, sp in zip(support_size, support_spacing)])
air_image = air_image - 1000 
sitk.WriteImage(air_image, "support_air.mhd")

sim = gate.Simulation()

## Paramètres globaux
sim.g4_verbose = False
sim.visu = False
sim.number_of_threads = 8
sim.output_dir = "./nema_final_sim"
sim.progress_bar = True
sim.check_volumes_overlap = True

sim.random_seed = 12345 + batch_id #todo : à valider ? 

mm = gate.g4_units.mm
cm = gate.g4_units.cm
keV = gate.g4_units.keV
sec = gate.g4_units.second
MBq = 1e6 * gate.g4_units.Bq

## Monde
sim.world.size = [1 * gate.g4_units.m] * 3
sim.world.material = "G4_AIR"


## Fantôme NEMA IEC natif
phantom = nema_p.add_iec_phantom(sim, "nema")
phantom.user_info.translation = [[0, 0, 0]]
rot_flip = R.from_euler('x', 180, degrees=True).as_matrix()
phantom.user_info.rotation = [rot_flip]

## Configuration SPECT
collimator_type = "megp"
spect, colli, crystal = spect_ge_nm670.add_spect_head(sim, "spect", "megp")
# Rayon de rotation (ROR) de 25cm + rotation circulaire
rad = 25 * cm
pos_x = rad * np.sin(np.radians(current_angle))
pos_z = rad * np.cos(np.radians(current_angle))

spect.user_info.translation = [[pos_x, 0, pos_z]]
rot_matrix = R.from_euler('y', 180 + current_angle, degrees=True).as_matrix()
spect.user_info.rotation = [rot_matrix]

## Digitizer
channels = [{"name": "peak208", "min": 192.4 * keV, "max": 223.6 * keV}]

hc = sim.add_actor("DigitizerHitsCollectionActor", f"Hits_{crystal.name}")
hc.attached_to = crystal.name
hc.attributes = ["EventID", "PostPosition", "TotalEnergyDeposit"]

cc = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{crystal.name}")
cc.attached_to = crystal.name
cc.input_digi_collection = hc.name
cc.channels = channels

proj = sim.add_actor("DigitizerProjectionActor", f"Projection_{crystal.name}")
proj.attached_to = crystal.name
proj.input_digi_collections = ["peak208"]
proj.spacing = [4.4 * mm, 4.4 * mm]
proj.size = [128, 128]
proj.output_filename = "projections_nema.mhd"

## Source Lu177 dans la sphère 6
# source = sim.add_source("GenericSource", "lu177_sphere")
# source.particle = "gamma"
# source.energy.type = "spectrum_discrete"
# source.energy.spectrum_energies = [208.36 * keV]
# source.energy.spectrum_weights = [1.0]
# source.attached_to = "nema_sphere_37mm"
# source.position.type = "sphere"
# source.position.radius = 18.5 * mm
# source.direction.type = "iso"
# source.activity = 1 * MBq / 100000

## Paramètres de l'activité (Concentration constante)
# On définit une activité de référence pour la plus grosse sphère (37mm)
activity_37mm = 1 * MBq / sim.number_of_threads
radius_ref = 18.5 * mm
vol_ref = (4/3) * np.pi * (radius_ref**3)
concentration = activity_37mm / vol_ref

# Liste des diamètres du fantôme NEMA IEC (en mm)
diameters = [10, 13, 17, 22, 28, 37]

## Boucle de création des sources
for d in diameters:
    r = (d / 2) * mm
    vol = (4/3) * np.pi * (r**3)
    sphere_activity = concentration * vol
    
    name = f"source_sphere_{d}mm"
    vol_target = f"nema_sphere_{d}mm"
    
    src = sim.add_source("GenericSource", name)
    src.particle = "gamma"
    # src.energy.type = "spectrum_discrete"
    # src.energy.spectrum_energies = [208.36 * keV]
    # src.energy.spectrum_weights = [1.0]
    src.energy.type = "mono"
    src.energy.mono = 208 * keV

    
    src.attached_to = vol_target
    src.position.type = "sphere"
    src.position.radius = r
    src.direction.type = "iso"
    src.activity = sphere_activity

## Physique
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"

## Run
sim.run_timing_intervals = [[0, 100 * sec]]
sim.run()