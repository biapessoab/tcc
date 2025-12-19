# ADNI: Script para extrair e renomear volumes NIfTI (.nii ou .nii.gz)
# da estrutura de pastas original do ADNI, adicionando o diagnóstico (DX).

import os
import pandas as pd
from datetime import datetime
import shutil
import glob

# Pasta raiz onde os dados do ADNI estão
root_dir = "./data/adni/ADNI"
# Pasta de destino onde os volumes processados e renomeados serão colocados
flat_dir = "./data/adni/adni-with-dx"
# Caminho para o arquivo CSV de diagnóstico
csv_path = "ADNIMERGE.csv"

# Diagnósticos permitidos
DIAGNOSTICOS_PERMITIDOS = ["Dementia", "CN"]

# Cria a pasta de destino se não existir
os.makedirs(flat_dir, exist_ok=True)

# --- 1. CARREGAR E PREPARAR CSV ---
try:
    df = pd.read_csv(csv_path, delimiter=",", encoding="utf-8", on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df["PTID"] = df["PTID"].astype(str)
    # Converte a data do exame para o formato datetime
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    print("✅ CSV carregado e preparado com sucesso!")
except Exception as e:
    print(f"❌ ERRO: Não foi possível carregar o CSV em {csv_path}. Verifique o caminho e o delimitador. Erro: {e}")
    exit()

def split_filename(file_name):
    """Divide o nome do arquivo e sua extensão (.nii ou .nii.gz)."""
    if file_name.endswith(".nii.gz"):
        return file_name[:-7], ".nii.gz"
    else:
        return os.path.splitext(file_name)

# --- 2. PROCESSAR E COPIAR ARQUIVOS ---
mantidos = 0
ignorados = 0

print(f"\n🔄 Iniciando a varredura em {root_dir}...")

# Percorre a estrutura de pastas do ADNI
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        # Filtra apenas por arquivos NIfTI
        if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
            continue

        full_path = os.path.join(dirpath, filename)

        # --- A. EXTRAIR PTID E DATA DO CAMINHO DA PASTA ---
        try:
            rel_path = os.path.relpath(full_path, root_dir)
            rel_parts = rel_path.split(os.sep)
            
            ptid = rel_parts[0]

            date_folder_candidates = [
                p for p in rel_parts if len(p) >= 10 and p[:10].count("-") == 2
            ]

            if not date_folder_candidates:
                raise ValueError("Nenhuma pasta de data encontrada")

            date_folder = date_folder_candidates[0]
            exam_date = datetime.strptime(date_folder[:10], "%Y-%m-%d")

        except Exception as e:
            ignorados += 1
            continue

        # --- B. BUSCAR DIAGNÓSTICO NO CSV ---
        # Busca a linha do CSV que corresponde ao PTID e Data do exame
        row = df[(df["PTID"] == ptid) & (df["EXAMDATE"] == exam_date)]
        
        if row.empty:
            ignorados += 1
            continue

        dx = str(row.iloc[0]["DX"]).strip()
        
        if dx not in DIAGNOSTICOS_PERMITIDOS:
            ignorados += 1
            continue
            
        # --- C. COPIAR E RENOMEAR ---
        
        # Renomeia o arquivo: [NomeOriginal]_[PTID]_[DataExame]_[DX].[ext]
        name, ext = split_filename(filename)
        exam_date_str = exam_date.strftime('%Y%m%d')
        
        new_filename = f"{name}_{ptid}_{exam_date_str}-{dx}{ext}"
        new_path = os.path.join(flat_dir, new_filename)

        shutil.copy2(full_path, new_path)
        mantidos += 1

# --- 3. RESUMO FINAL ---
print("\n" + "="*30)
print("📊 RESUMO FINAL DO PROCESSAMENTO")
print("="*30)
print(f"✔️ Volumes válidos encontrados e copiados (AD/CN): {mantidos}")
print(f"❌ Volumes ignorados (outros DX, sem data ou sem match): {ignorados}")
print(f"📂 Total de arquivos na pasta de destino ({os.path.basename(flat_dir)}): {len(glob.glob(os.path.join(flat_dir, '*.nii*')))}")
print("Lembre-se: Arquivos anteriores foram mantidos.")
print("="*30)