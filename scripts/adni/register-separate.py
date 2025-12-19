import os
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path

base_dir = "./data/adni/adni-stripped"
imagem_ref = os.path.join(
    os.environ["FSLDIR"],
    "data/standard/MNI152_T1_2mm_brain.nii.gz"
)

output_dirs = {
    'original': './data/adni/adni-original-registered',
    'translacao': './data/adni/adni-translacao-only',
    'rotacao': './data/adni/adni-rotacao-only',
    'escala': './data/adni/adni-escala-only',
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Diretório temporário para matrizes
temp_mat_dir = "./temp_matrices"
os.makedirs(temp_mat_dir, exist_ok=True)


# VALIDAÇÃO DE DIMENSÕES

def validar_imagem_registrada(img_path, expected_shape=(91, 109, 91)):
    """
    Valida se a imagem registrada está correta
    Retorna (valida, info)
    """
    try:
        img = nib.load(img_path)
        data = img.get_fdata()
        
        # Verificar dimensões
        if data.shape != expected_shape:
            return False, f"Dimensão incorreta: {data.shape} (esperado: {expected_shape})"
        
        # Verificar se tem voxels válidos
        n_voxels = np.sum(data > 0)
        if n_voxels < 1000:
            return False, f"Poucos voxels: {n_voxels}"
        
        # Verificar por NaN ou Inf
        if np.any(np.isnan(data)):
            return False, "Contém NaN"
        
        if np.any(np.isinf(data)):
            return False, "Contém Inf"
        
        return True, f"OK ({n_voxels} voxels)"
        
    except Exception as e:
        return False, f"Erro ao validar: {str(e)}"


# REGISTRO COM FLIRT

def fazer_registro_completo(imagem_in, imagem_ref, mat_out, img_out):
    """
    Faz registro completo gerando TANTO a matriz QUANTO a imagem registrada
    """
    try:
        cmd = [
            "flirt",
            "-in", imagem_in,
            "-ref", imagem_ref,
            "-out", img_out,        # gera a imagem registrada
            "-omat", mat_out,       #salva a matriz
            "-cost", "mutualinfo",
            "-dof", "12",           # 12 graus de liberdade (afim completo)
            "-interp", "trilinear"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, "Sucesso"
        
    except subprocess.CalledProcessError as e:
        return False, f"FLIRT falhou: {e.stderr}"
    except Exception as e:
        return False, f"Erro: {str(e)}"


def aplicar_transformacao(imagem_in, imagem_ref, matriz_file, output_file):
    """
    Aplica uma transformação específica usando FLIRT
    """
    try:
        cmd = [
            "flirt",
            "-in", imagem_in,
            "-ref", imagem_ref,
            "-out", output_file,
            "-init", matriz_file,
            "-applyxfm",
            "-interp", "trilinear"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, "Sucesso"
        
    except subprocess.CalledProcessError as e:
        return False, f"FLIRT falhou: {e.stderr}"
    except Exception as e:
        return False, f"Erro: {str(e)}"


# DECOMPOSIÇÃO DE MATRIZ AFIM

def ler_matriz_afim(mat_file):
    """Lê a matriz de transformação afim do FLIRT"""
    return np.loadtxt(mat_file)


def decompor_matriz_afim(matriz):
    """
    Decompõe matriz afim em componentes básicos usando decomposição SVD
    """
    # Extrair partes
    A = matriz[:3, :3]  # Parte linear (rotação + escala + shear)
    t = matriz[:3, 3]    # Translação
    
    # Decomposição SVD para separar rotação e escala
    U, S, Vt = np.linalg.svd(A)
    
    # Rotação pura (ortogonal)
    R = U @ Vt
    
    # Garantir que é uma rotação (det = 1, não reflexão)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    # Escala (valores singulares)
    escala = S
    
    return {
        'translacao': t,
        'rotacao': R,
        'escala': escala,
        'matriz_completa': A
    }


def criar_matriz_identidade():
    """Cria matriz identidade 4x4"""
    return np.eye(4)


def criar_matriz_por_tipo(componentes, tipo):
    """
    Cria matriz de transformação baseada no tipo especificado
    """
    M = criar_matriz_identidade()
    
    if tipo == 'translacao':
        # Apenas translação
        M[:3, 3] = componentes['translacao']
        
    elif tipo == 'rotacao':
        # Apenas rotação
        M[:3, :3] = componentes['rotacao']
        
    elif tipo == 'escala':
        # Apenas escala
        M[:3, :3] = np.diag(componentes['escala'])
        
    elif tipo == 'original':
        # Transformação completa (já foi aplicada, não precisa refazer)
        M[:3, :3] = componentes['matriz_completa']
        M[:3, 3] = componentes['translacao']
    
    return M


def salvar_matriz(matriz, arquivo):
    """Salva matriz no formato do FLIRT"""
    np.savetxt(arquivo, matriz, fmt='%.10f')


# PROCESSAMENTO PRINCIPAL

def processar_imagem_completo(imagem_in, imagem_ref, nome_base):
    """
    Processa uma imagem:
    1. Faz registro completo (gera imagem + matriz)
    2. Valida a imagem registrada
    3. Decompõe a matriz e gera variantes
    """
    print(f"\n  {'='*70}")
    print(f"  Processando: {nome_base}")
    print(f"  {'='*70}")
    
    resultados = {
        'nome': nome_base,
        'sucesso': False,
        'original': None,
        'translacao': None,
        'rotacao': None,
        'escala': None,
        'erro': None
    }
    
    mat_file = os.path.join(temp_mat_dir, f"{nome_base}.mat")
    img_original = os.path.join(output_dirs['original'], f"{nome_base}.nii.gz")
    
    # ETAPA 1: Registro completo
    print(f"  [1/4] Registro completo...", end=" ")
    
    if os.path.exists(img_original) and os.path.exists(mat_file):
        # Verificar se já existe e está válido
        valida, info = validar_imagem_registrada(img_original)
        if valida:
            print(f"✓ (já existe e válido)")
            resultados['original'] = img_original
        else:
            print(f"⚠️ Existe mas inválido: {info}")
            print(f"      Refazendo registro...", end=" ")
            sucesso, msg = fazer_registro_completo(imagem_in, imagem_ref, mat_file, img_original)
            if sucesso:
                print("✓")
                resultados['original'] = img_original
            else:
                print(f"✗ {msg}")
                resultados['erro'] = f"Registro falhou: {msg}"
                return resultados
    else:
        # Fazer registro novo
        sucesso, msg = fazer_registro_completo(imagem_in, imagem_ref, mat_file, img_original)
        if sucesso:
            print("✓")
            resultados['original'] = img_original
        else:
            print(f"✗ {msg}")
            resultados['erro'] = f"Registro falhou: {msg}"
            return resultados
    
    # ETAPA 2: Validar imagem registrada
    print(f"  [2/4] Validando registro...", end=" ")
    valida, info = validar_imagem_registrada(img_original)
    if not valida:
        print(f"✗ {info}")
        resultados['erro'] = f"Validação falhou: {info}"
        return resultados
    print(f"✓ {info}")
    
    # ETAPA 3: Decompor matriz
    print(f"  [3/4] Decompondo transformação...", end=" ")
    try:
        matriz = ler_matriz_afim(mat_file)
        componentes = decompor_matriz_afim(matriz)
        print("✓")
        
        # Mostrar componentes
        print(f"      Translação: X={componentes['translacao'][0]:6.2f}, Y={componentes['translacao'][1]:6.2f}, Z={componentes['translacao'][2]:6.2f} mm")
        print(f"      Escala:     X={componentes['escala'][0]:6.3f}, Y={componentes['escala'][1]:6.3f}, Z={componentes['escala'][2]:6.3f}")
        
        # Calcular ângulos de rotação
        R = componentes['rotacao']
        rot_x = np.arctan2(R[2,1], R[2,2]) * 180/np.pi
        rot_y = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2)) * 180/np.pi
        rot_z = np.arctan2(R[1,0], R[0,0]) * 180/np.pi
        print(f"      Rotação:    X={rot_x:6.2f}°, Y={rot_y:6.2f}°, Z={rot_z:6.2f}°")
        
    except Exception as e:
        print(f"✗ {e}")
        resultados['erro'] = f"Decomposição falhou: {e}"
        return resultados
    
    # ETAPA 4: Gerar transformações isoladas
    print(f"  [4/4] Gerando transformações isoladas:")
    
    for tipo in ['translacao', 'rotacao', 'escala']:
        print(f"      → {tipo:12s}...", end=" ")
        
        try:
            # Criar matriz específica
            matriz_tipo = criar_matriz_por_tipo(componentes, tipo)
            
            # Caminhos de saída
            mat_tipo = os.path.join(output_dirs[tipo], f"{nome_base}.mat")
            img_tipo = os.path.join(output_dirs[tipo], f"{nome_base}.nii.gz")
            
            # Verificar se já existe e está válido
            if os.path.exists(img_tipo):
                valida, info = validar_imagem_registrada(img_tipo)
                if valida:
                    print(f"✓ (já existe)")
                    resultados[tipo] = img_tipo
                    continue
            
            # Salvar matriz
            salvar_matriz(matriz_tipo, mat_tipo)
            
            # Aplicar transformação
            sucesso, msg = aplicar_transformacao(imagem_in, imagem_ref, mat_tipo, img_tipo)
            
            if sucesso:
                # Validar resultado
                valida, info = validar_imagem_registrada(img_tipo)
                if valida:
                    print(f"✓")
                    resultados[tipo] = img_tipo
                else:
                    print(f"⚠️ Gerado mas inválido: {info}")
            else:
                print(f"✗ {msg}")
                
        except Exception as e:
            print(f"✗ Erro: {e}")
    
    resultados['sucesso'] = True
    return resultados

def main():
    print("\n" + "="*80)
    print("GERADOR DE DATASETS COM TRANSFORMAÇÕES SEPARADAS")
    print("COM VALIDAÇÃO E GARANTIA DE DIMENSÕES CORRETAS")
    print("="*80)
    print("\nEste script:")
    print("  1. Faz registro completo (FLIRT) gerando imagem registrada + matriz")
    print("  2. Valida que cada imagem ficou em 91×109×91 voxels (MNI152)")
    print("  3. Decompõe a transformação em componentes isolados")
    print("  4. Gera datasets com transformações específicas")
    print("\nDatasets gerados:")
    print("  • original: registro completo (12 DOF)")
    print("  • translacao: apenas translação")
    print("  • rotacao: apenas rotação")
    print("  • escala: apenas escala")
    print("\n" + "="*80 + "\n")
    
    # Buscar todas as imagens
    todas_imagens = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                todas_imagens.append(os.path.join(root, file))
    
    total = len(todas_imagens)
    print(f"📁 Encontradas {total} imagens para processar.\n")
    
    if total == 0:
        print("❌ Nenhuma imagem encontrada!")
        return
    
    print(f"📋 Referência: {imagem_ref}\n")
    ref_img = nib.load(imagem_ref)
    print(f"   Dimensões: {ref_img.shape}")
    print(f"   Voxel size: {ref_img.header.get_zooms()[:3]}")
    print()
    
    print("="*80)
    print("PROCESSAMENTO")
    print("="*80)
    
    resultados_todos = []
    sucessos = 0
    falhas = 0
    
    for idx, imagem_in in enumerate(todas_imagens, start=1):
        nome_base = os.path.basename(imagem_in).replace(".nii.gz", "").replace(".nii", "")
        
        print(f"\n[{idx}/{total}] {nome_base}")
        
        resultado = processar_imagem_completo(imagem_in, imagem_ref, nome_base)
        resultados_todos.append(resultado)
        
        if resultado['sucesso']:
            sucessos += 1
        else:
            falhas += 1
            print(f"  ❌ FALHOU: {resultado['erro']}")
    
    print("\n" + "="*80)
    print("RESUMO FINAL")
    print("="*80)
    
    print(f"\n✅ Sucessos: {sucessos}/{total}")
    print(f"❌ Falhas:   {falhas}/{total}")
    
    if falhas > 0:
        print(f"\n⚠️  Imagens que falharam:")
        for r in resultados_todos:
            if not r['sucesso']:
                print(f"   • {r['nome']}: {r['erro']}")
    
    print(f"\n📁 Datasets gerados:")
    for tipo, caminho in output_dirs.items():
        arquivos = [f for f in os.listdir(caminho) if f.endswith('.nii.gz')]
        num_arquivos = len(arquivos)
        
        exemplos_validos = 0
        for exemplo in arquivos[:5]:  # Validar 5 exemplos
            valida, _ = validar_imagem_registrada(os.path.join(caminho, exemplo))
            if valida:
                exemplos_validos += 1
        
        status = "✓" if exemplos_validos == min(5, num_arquivos) else "⚠️"
        print(f"  {status} {tipo:12s}: {num_arquivos:3d} imagens → {caminho}")
    
    print("\n" + "="*80)
    print("✅ PROCESSAMENTO CONCLUÍDO!")
    print("="*80)
    print("\n💡 Próximo passo: Execute o script de métricas para validar os resultados")
    print("   python scripts/adni/register-statistic.py")
    print()


if __name__ == "__main__":
    main()