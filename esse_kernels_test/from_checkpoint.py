import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FOLDER = "./output"
OF_2 = "./output_test"
NB_RUNS = 20
KRNL_SIZE_XY = 64      # Taille spatiale X et Y
KRNL_SIZE_Z = 64

checkpoint_path = os.path.join(OUTPUT_FOLDER, "esse_kernels_checkpoint.npy")
print(f"Checkpoint path: {checkpoint_path}")
print(os._exists(checkpoint_path))

if os.path.exists(checkpoint_path):
    print(f">>> Chargement du checkpoint : {checkpoint_path}")
    cp = np.load(checkpoint_path, allow_pickle=True).item()
    
    # final_kernels = cp['kernels']
    # amu_kernels_accumulation = cp['amu_accum']
    norm_factor = cp['norm_factor']
    
    last_run = cp['last_run']
    k0_accum = cp['k0']
    amu_num_accum = cp['amu_num']

    print(f"Checkpoint chargé : Run {last_run}")

print("\n>>> Calcul des 3 composantes ESSE (k0, k1, k2)...")
# 1. Normalisation du kernel de base (Ordre 0)
k0_final = k0_accum / norm_factor

# 2. Calcul du ratio a_mu (sécurisé contre la division par zéro)
a_mu = np.divide(amu_num_accum, k0_accum, out=np.zeros_like(k0_accum), where=k0_accum != 0)

# 3. Création des ordres 1 et 2 selon Taylor
k1_final = k0_final * a_mu
k2_final = k0_final * (a_mu ** 2)

# 4. Assemblage du tenseur 4D final pour PyTorch: shape = [3, Z, Y, X]
esse_tensor_4d = np.stack([k0_final, k1_final, k2_final], axis=0)

# --- SAUVEGARDE ET DIAGNOSTIC ---
np.save(os.path.join(OF_2, "esse_kernels_ready_for_pytorch.npy"), esse_tensor_4d)
print(f"\n✅ Tenseur ESSE 4D généré avec succès ! Shape : {esse_tensor_4d.shape}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(k0_final[KRNL_SIZE_Z//2, :, :], cmap='hot')
plt.title("k0 (Base Scatter)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(k1_final[KRNL_SIZE_Z//2, :, :], cmap='magma')
plt.title("k1 (Delta Mu)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(k2_final[KRNL_SIZE_Z//2, :, :], cmap='viridis')
plt.title("k2 (Delta Mu ^2)")
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(OF_2, "diag_esse_3d.png"))


