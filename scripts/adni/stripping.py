import os
import nibabel as nib
import matplotlib.pyplot as plt
import subprocess

# 1. Caminhos de entrada/saída
orig_path = "data/adni/adni-with-dx/ADNI_067_S_0059_20250721190207011-CN.nii"

out_dir = "./debug_single"
os.makedirs(out_dir, exist_ok=True)

reoriented_path = os.path.join(out_dir, "image_reoriented.nii.gz")
stripped_path   = os.path.join(out_dir, "image_stripped.nii.gz")
mni_registered_path = os.path.join(out_dir, "image_MNI.nii.gz")

# Caminho para o atlas do FSL
mni_atlas = os.path.join(
    os.environ["FSLDIR"],
    "data/standard/MNI152_T1_2mm_brain.nii.gz"
)

# 2. Reorienta 
subprocess.run(["fslreorient2std", orig_path, reoriented_path], check=True)

# 3. Skull stripping
subprocess.run([
    "bet", reoriented_path, stripped_path,
    "-R", "-f", "0.5", "-g", "-0.2"
], check=True)

# 4. Registro para MNI152 (FLIRT)
subprocess.run([
    "flirt",
    "-in", stripped_path,
    "-ref", mni_atlas,
    "-out", mni_registered_path,
    "-omat", os.path.join(out_dir, "affine.mat"),
    "-dof", "12"
], check=True)

# 5. Carrega volumes
orig_img = nib.load(orig_path).get_fdata()
reor_img = nib.load(reoriented_path).get_fdata()
strip_img = nib.load(stripped_path).get_fdata()
mni_img = nib.load(mni_registered_path).get_fdata()

# 6. Função da fatia coronal
def get_central_coronal(volume):
    c = volume.shape[1] // 2
    return volume[:, c, :]

s_orig  = get_central_coronal(orig_img)
s_reor  = get_central_coronal(reor_img)
s_strip = get_central_coronal(strip_img)
s_mni   = get_central_coronal(mni_img)

# 7. Plot 
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(s_orig.T, cmap="gray", origin="lower")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Reoriented (RAS)")
plt.imshow(s_reor.T, cmap="gray", origin="lower")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Stripped")
plt.imshow(s_strip.T, cmap="gray", origin="lower")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("MNI Registered")
plt.imshow(s_mni.T, cmap="gray", origin="lower")
plt.axis("off")

plt.tight_layout()
plt.show()
