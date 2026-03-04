import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Chargement du fichier
file_path = "esse_kernels_3d.npy"

if not os.path.exists(file_path):
    print(f"Erreur : Le fichier {file_path} est introuvable.")
else:
    kernels = np.load(file_path)
    
    # 2. Vérification des dimensions
    # Format attendu : (NB_SLICES, KRNL_SIZE, KRNL_SIZE)
    print(f"Dimensions de la matrice : {kernels.shape}")
    print(f"Valeur max globale : {np.max(kernels):.4e}")
    print(f"Nombre de tranches : {kernels.shape[0]}")

    # 3. Visualisation de quelques tranches
    # On affiche la première, la moitié et la dernière
    slices_to_show = [0, kernels.shape[0]//2, kernels.shape[0]-1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, idx in enumerate(slices_to_show):
        im = axes[i].imshow(kernels[idx], cmap='hot', interpolation='nearest')
        axes[i].set_title(f"Tranche {idx}\n(Profondeur {(idx+0.5)*0.48:.2f} cm)")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.show()

    # 4. Si vous voulez voir les valeurs numériques brutes de la zone centrale
    mid = kernels.shape[1] // 2
    print("\nValeurs au centre du dernier kernel (5x5 pixels) :")
    print(kernels[-1, mid-2:mid+3, mid-2:mid+3])