import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
file_path = "./output/esse_kernels_3d.npy"
output_plots_dir = "./output/plots_kernels"
os.makedirs(output_plots_dir, exist_ok=True)

if not os.path.exists(file_path):
    print(f"Erreur : Le fichier {file_path} est introuvable.")
else:
    kernels = np.load(file_path)
    
    num_slices = kernels.shape[1]
    print(f"Dimensions de la matrice : {kernels.shape}")
    print(f"Nombre de tranches à traiter : {num_slices}")

    # Fixer la valeur max pour avoir la même échelle de couleur sur toutes les images
    global_max = np.max(kernels)

    for i in range(num_slices):
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Affichage de la coupe i
        # On utilise global_max pour que l'intensité soit comparable visuellement
        im = ax.imshow(kernels[:, i, :], cmap='hot', interpolation='nearest', vmin=0, vmax=global_max)
        
        ax.set_xlabel("Position Y")
        ax.set_ylabel("Position X")
        ax.set_title(f"Kernel ESSE - Coupe {i}\n(Z Simulation)")
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        # Enregistrement
        save_path = os.path.join(output_plots_dir, f"kernel_slice_{i:02d}.png")
        plt.savefig(save_path)
        plt.close(fig) # Important pour libérer la mémoire vive
        
        if (i + 1) % 5 == 0:
            print(f"Tranche {i+1}/{num_slices} enregistrée...")

    print(f"\nTerminé. Toutes les images sont dans : {output_plots_dir}")