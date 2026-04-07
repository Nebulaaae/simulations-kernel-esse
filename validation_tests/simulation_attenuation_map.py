import opengate as gate
import opengate.contrib.phantoms.nemaiec as nema_p
from opengate.voxelize import voxelize_geometry, write_voxelized_geometry
from scipy.spatial.transform import Rotation as R
import os

# --- Configuration ---
output_dir = "./nema_maps"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sim = gate.Simulation()

# Unités
mm = gate.g4_units.mm
cm = gate.g4_units.cm
m = gate.g4_units.m
keV = gate.g4_units.keV

# Monde
sim.world.size = [2 * m] * 3
sim.world.material = "G4_AIR"

# Fantôme NEMA IEC
phantom = nema_p.add_iec_phantom(sim, "nema")
phantom.user_info.translation = [[0, 0, 0]]
# rot_flip = R.from_euler('x', 180, degrees=True).as_matrix()
# phantom.user_info.rotation = [rot_flip]

# --- Voxelisation ---

# iec_plastic = {
#     "name": "IEC_PLASTIC",
#     "density": 1.18 * (gate.g4_units.g / gate.g4_units.cm3),
#     "elements": ["C", "H", "O"],
#     "weights": [0.5998, 0.0805, 0.3197]
# }
# sim.volume_manager.user_info.materials.append(iec_plastic)

volume_labels, image = voxelize_geometry(sim, extent=phantom, spacing=(1.0*mm, 1.0*mm, 1.0*mm))
voxel_mhd_path = os.path.join(output_dir, "nema.mhd")
filenames = write_voxelized_geometry(sim, volume_labels, image, voxel_mhd_path)
print(f"Voxelisation terminée. Fichiers : {filenames}")

for v in [name for name in sim.volume_manager.volumes.keys() if name.startswith("nema")]:
    sim.volume_manager.volumes.pop(v)




# voxel_json_path = os.path.join(output_dir, "nema_labels.json")
# patient.set_materials_from_voxelisation(voxel_json_path)

sim.volume_manager.add_material_database(os.path.join(output_dir, "nema.db"))
patient = sim.add_volume("ImageVolume", "patient_vox")
patient.image = voxel_mhd_path
patient.read_label_to_material(os.path.join(output_dir, "nema_labels.json"))

# --- Génération de la Mu-Map ---

mumap = sim.add_actor("AttenuationImageActor", "attenuation_map")
mumap.image_volume = patient
mumap.output_filename = os.path.join(output_dir, "nema_mu_map_208keV.mhd")
mumap.energy = 208 * keV # Énergie du pic principal du Lu177
mumap.database = "NIST"


mumap.attenuation_image.active = True
mumap.attenuation_image.write_to_disk = True

# Initialisation et exécution
sim.run()

print(f"\nTerminé. Les fichiers sont dans : {output_dir}")