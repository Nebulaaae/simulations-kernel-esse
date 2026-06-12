import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FOLDER = "./output"
OF_2 = "./output_test"
NB_SLICES = 9
RUNS_PER_SLICE = 3

checkpoint_path = os.path.join(OUTPUT_FOLDER, "esse_kernels_checkpoint.npy.tmp.npy")
print(f"Checkpoint path: {checkpoint_path}")
print(os._exists(checkpoint_path))

if os.path.exists(checkpoint_path):
    print(f">>> Chargement du checkpoint : {checkpoint_path}")
    cp = np.load(checkpoint_path, allow_pickle=True).item()
    
    final_kernels = cp['kernels']
    amu_kernels_accumulation = cp['amu_accum']
    norm_factor = cp['norm_factor']
    
    last_run_idx = cp['last_run']
    last_slice_idx = cp['last_slice']

    print(f"Checkpoint chargé : Slice {last_slice_idx}, Run {last_run_idx}")

    print("\n>>> Calcul final des kernels amu (a_mu)...")
final_amu_kernels = np.divide(
    amu_kernels_accumulation, 
    final_kernels, 
    out=np.zeros_like(amu_kernels_accumulation), 
    where=final_kernels != 0
)
os.makedirs(OF_2, exist_ok=True)

print("\n>>> Génération de l'image de diagnostic : diag_esse.png")
plt.figure(figsize=(15, 5))
# Affichage du Kernel de base (Log scale pour voir les queues de diffusion)
plt.subplot(1, 3, 1)
plt.imshow(final_kernels[:, NB_SLICES//2, :], cmap='hot')
plt.title("Kernel Diffusion (Poids)")
plt.colorbar()

# Affichage du Numérateur (pour voir s'il y a de l'info avant division)
plt.subplot(1, 3, 2)
plt.imshow(amu_kernels_accumulation[:, NB_SLICES//2, :], cmap='magma')
plt.title("Numérateur Delta Mu")
plt.colorbar()

# Affichage du Ratio Final (amu)
plt.subplot(1, 3, 3)
plt.imshow(final_amu_kernels[:, NB_SLICES//2, :], cmap='viridis')
plt.title("Kernel Delta Mu Final")
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(OF_2, "diag_esse.png"))
print(f"Image sauvegardée dans : {os.path.join(OF_2, 'diag_esse.png')}")

# --- NORMALISATION FINALE ET SAUVEGARDE ---
if norm_factor > 0:
    final_kernels /= (norm_factor)
    np.save(os.path.join(OF_2, "esse_kernels_3d.npy"), final_kernels)
    np.save(os.path.join(OF_2, "esse_amu_kernels_3d.npy"), final_amu_kernels)
    print("\nKernels ESSE 3D générés et normalisés avec succès.")
else:
    print("\nErreur : Normalisation impossible (norm_factor = 0)")


