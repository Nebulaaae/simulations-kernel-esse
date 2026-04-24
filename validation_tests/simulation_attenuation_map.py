import opengate as gate
import opengate.contrib.phantoms.nemaiec as nema_p
from opengate.voxelize import voxelize_geometry, write_voxelized_geometry
from scipy.spatial.transform import Rotation as R
import numpy as np
import SimpleITK as sitk
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
sim.check_volumes_overlap = True

# Fantôme NEMA IEC
# phantom = nema_p.add_iec_phantom(sim, "nema")
# phantom.user_info.translation = [[0, 0, 0]]
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
import SimpleITK as sitk
import numpy as np
import itk

IMG_SIZE = 128
PIXEL_SIZE = 0.44 

# 1. Définition de la taille (ex: 563.2 mm pour couvrir 128 pixels de 4.4 mm)
size_mm = IMG_SIZE * PIXEL_SIZE * 10 

# 2. Création d'une boîte virtuelle pour définir l'étendue
# On l'ajoute au monde, centrée par défaut à [0,0,0]
container = sim.add_volume("Box", "external_box")
container.size = [size_mm] * 3
container.material = "G4_AIR" # Important : Air pour ne pas fausser la mu-map
container.translation = [0, 0, 0]

# 2. Créer le Fantôme comme ENFANT de la boîte
# On utilise l'argument 'mother'
phantom = nema_p.add_iec_phantom(sim, "nema")
phantom.user_info.translation = [[0, 0, 0]] 
phantom.mother = "external_box"

volume_labels, image_itk = voxelize_geometry(
    sim, 
    extent=container, 
    spacing=(1.0*mm, 1.0*mm, 1.0*mm)
)

array = itk.array_from_image(image_itk)
image_sitk = sitk.GetImageFromArray(array)

spacing_itk = image_itk.GetSpacing()
image_sitk.SetSpacing([spacing_itk[0], spacing_itk[1], spacing_itk[2]])

# Calcul du centrage strict
size = np.array(image_sitk.GetSize())
spacing = np.array(image_sitk.GetSpacing())
new_origin = -(size * spacing) / 2.0
image_sitk.SetOrigin(new_origin)

# Écriture
voxel_mhd_path = os.path.join(output_dir, "nema.mhd")
sitk.WriteImage(image_sitk, voxel_mhd_path)

print(f"Voxelisation terminée avec origine centrée : {image_sitk.GetOrigin()}")

for v in [name for name in sim.volume_manager.volumes.keys() if name.startswith("nema") or name.startswith("external_box")]:
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