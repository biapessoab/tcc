import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

input_dir = "/Users/anabeatrizbraz/Documents/PUC/tcc/data/adni/adni-registered"
output_dir = "/Users/anabeatrizbraz/Documents/PUC/tcc/data/adni/adni-slices"

os.makedirs(output_dir, exist_ok=True)

# Percorre todos os arquivos .nii
for file in os.listdir(input_dir):
    if file.endswith(".nii") or file.endswith(".nii.gz"):
        filepath = os.path.join(input_dir, file)
        print(f"Processando {filepath}...")

        # Carrega o volume
        img = nib.load(filepath)
        data = img.get_fdata()

        # Normaliza para 0-255
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        data = data.astype(np.uint8)

        # Pega fatia coronal central (eixo Y = 1)
        mid_idx = data.shape[1] // 2
        slice_img = data[:, mid_idx, :]

        # Nome base do arquivo
        vol_name = os.path.splitext(file)[0]
        output_path = os.path.join(output_dir, f"{vol_name}_coronal.png")

        # Salva imagem
        plt.imsave(output_path, np.rot90(slice_img), cmap="gray")
