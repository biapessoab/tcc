# ADNI
import os
import subprocess
import time

# Pasta com todas as imagens
base_dir = "./data/adni/adni-stripped"

# Atlas de referência
imagem_ref = os.path.join(
    os.environ["FSLDIR"],
    "data/standard/MNI152_T1_2mm_brain.nii.gz"
)

output_dir = "./data/adni/adni-registered"
os.makedirs(output_dir, exist_ok=True)
  
def pegar_resolucao(imagem):
    """Usa fslinfo para pegar a resolução dos voxels"""
    try:
        result = subprocess.run(["fslinfo", imagem], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        pixdim = {line.split()[0]: line.split()[1] for line in lines if line.startswith("pixdim")}
        return f"{pixdim['pixdim1']} x {pixdim['pixdim2']} x {pixdim['pixdim3']} mm"
    except Exception as e:
        return f"Erro ao obter resolução: {e}"

print("\n=== Novo registro iniciado em " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===")
print("Registro de imagens com FLIRT + FNIRT")
print("="*60 + "\n")

# Lista todas as imagens
todas_imagens = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            todas_imagens.append(os.path.join(root, file))

total = len(todas_imagens)

for idx, imagem_in in enumerate(todas_imagens, start=1):
    file = os.path.basename(imagem_in)
    print(f"\n[{idx}/{total}] Processando {file}...")

    try:
        nome_base = file.replace(".nii.gz", "").replace(".nii", "")
        caminho_flirt_out = os.path.join(output_dir, f"{nome_base}_affine.nii.gz")
        caminho_mat = os.path.join(output_dir, f"{nome_base}_affine.mat")
        # caminho_fnirt_out = os.path.join(output_dir, f"{nome_base}_nonlin.nii.gz")
        caminho_warp = os.path.join(output_dir, f"{nome_base}_warpcoef.nii.gz")

        if os.path.exists(caminho_flirt_out):
            print(f"⏭️ {file} já processado. Pulando.")
            continue

        res_in = pegar_resolucao(imagem_in)

        # --- Registro Linear (FLIRT) ---
        if not os.path.exists(caminho_flirt_out):
            cmd_flirt = [
                "flirt",
                "-in", imagem_in,
                "-ref", imagem_ref,
                "-out", caminho_flirt_out,
                "-omat", caminho_mat,
                "-cost", "mutualinfo"
            ]
            start_flirt = time.time()
            print(f"   ➡️ Registrando linear (FLIRT)...")
            subprocess.run(cmd_flirt, check=True)
            end_flirt = time.time()
            tempo_flirt = end_flirt - start_flirt
            res_flirt = pegar_resolucao(caminho_flirt_out)
        else:
            print(f"   ⏭️ FLIRT já existe. Pulando.")
            tempo_flirt = 0
            res_flirt = pegar_resolucao(caminho_flirt_out)

        print(f"   ✅ Concluído: FLIRT {tempo_flirt:.2f}s")
        print(f"   Resolução original: {res_in}")
        print(f"   FLIRT: {res_flirt}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Falha ao registrar {file}: {e}")

print("\n✔️ Registro completo para todas as imagens.\n")
