import numpy as np
import matplotlib.pyplot as plt
import os

# Chargement du fichier
file_path = "./output/esse_kernels.npy"

if not os.path.exists(file_path):
    print(f"Erreur : Le fichier {file_path} est introuvable.")
else:
    kernels = np.load(file_path)
    
    print(f"Dimensions de la matrice : {kernels.shape}")
    print(f"Valeur max globale : {np.max(kernels):.4e}")
    print(f"Nombre de tranches : {kernels.shape[0]}")

    slice_to_show = 0
    
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    
    # Dans votre script de plot
    im = axes.imshow(kernels[0], cmap='hot', interpolation='nearest')
    axes.set_xlabel("Position Y (Transversal)")
    axes.set_ylabel("Position Z (Profondeur vers détecteur)")
    axes.set_title(f"Position de dernière diffusion des photons détectés\n")
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig("img.png")

    mid = kernels.shape[1] // 2
    print("\nValeurs au centre du dernier kernel (5x5 pixels) :")
    print(kernels[-1, mid-2:mid+3, mid-2:mid+3])