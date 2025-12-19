import os
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, ttest_rel
from nibabel.processing import resample_from_to

# MÉTRICAS

def calcular_informacao_mutua_normalizada(img1, img2, bins=32):
    mask = (img1 > 0) & (img2 > 0)
    img1_masked = img1[mask]
    img2_masked = img2[mask]

    if len(img1_masked) == 0:
        return np.nan 

    hist_2d, _, _ = np.histogram2d(img1_masked.ravel(), img2_masked.ravel(), bins=bins)

    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_nz = px[px > 0]
    py_nz = py[py > 0]
    pxy_nz = pxy[pxy > 0]

    H_x = -np.sum(px_nz * np.log2(px_nz))
    H_y = -np.sum(py_nz * np.log2(py_nz))
    H_xy = -np.sum(pxy_nz * np.log2(pxy_nz))

    return (H_x + H_y) / H_xy if H_xy > 0 else np.nan


def calcular_correlacao_cruzada_normalizada(img1, img2):
    mask = (img1 > 0) & (img2 > 0)
    img1_masked = img1[mask].ravel()
    img2_masked = img2[mask].ravel()

    if len(img1_masked) < 2:
        return np.nan

    corr, _ = pearsonr(img1_masked, img2_masked)
    return corr


def calcular_erro_medio_quadratico(img1, img2):
    mask = (img1 > 0) & (img2 > 0)
    if np.sum(mask) == 0:
        return np.nan
    return np.mean((img1[mask] - img2[mask]) ** 2)


def calcular_erro_absoluto_medio(img1, img2):
    mask = (img1 > 0) & (img2 > 0)
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs(img1[mask] - img2[mask]))


# VALIDAÇÃO DE DIMENSÕES
def validar_dimensoes(img_path, expected_shape=(91, 109, 91)):
    """
    Valida se a imagem tem as dimensões esperadas (MNI152)
    Retorna True se válida, False caso contrário
    """
    try:
        img = nib.load(img_path)
        data = img.get_fdata()
        
        if data.shape != expected_shape:
            return False, data.shape
        
        # Verificar se tem voxels válidos
        if np.sum(data > 0) < 1000: 
            return False, f"Poucos voxels: {np.sum(data > 0)}"
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False, "Contém NaN/Inf"
        
        return True, data.shape
        
    except Exception as e:
        return False, f"Erro: {str(e)}"


# COHEN'S D (TAMANHO DE EFEITO)

def calcular_cohens_d(grupo1, grupo2):
    """
    Calcula Cohen's d para amostras pareadas.
    Remove NaN antes do cálculo.
    """
    # Criar série válida (sem NaN)
    mask = ~(np.isnan(grupo1) | np.isnan(grupo2))
    g1_valid = grupo1[mask]
    g2_valid = grupo2[mask]
    
    if len(g1_valid) == 0:
        return np.nan
    
    diferenca = g1_valid - g2_valid
    s = np.std(diferenca, ddof=1)
    d = np.mean(diferenca) / s if s > 0 else 0
    return d


def interpretar_cohens_d(d):
    """Retorna interpretação textual do Cohen's d"""
    if np.isnan(d):
        return "não calculável"
    d_abs = abs(d)
    if d_abs < 0.2:
        return "pequeno"
    elif d_abs < 0.5:
        return "pequeno a médio"
    elif d_abs < 0.8:
        return "médio"
    else:
        return "grande"


# RESAMPLE COMPATÍVEL COM FLIRT
def resample_to_reference_FLIRT(img_reg, img_ref):
    """
    Reamostra img_reg para o espaço da img_ref usando nibabel.processing.resample_from_to,
    equivalente à interpolação do FLIRT (trilinear).
    """
    try:
        resampled = resample_from_to(img_reg, img_ref)
        return resampled.get_fdata()
    except Exception as e:
        print(f"\n     ⚠️ Erro no resample: {e}")
        raise


# AVALIAÇÃO DO REGISTRO

def avaliar_registro(img_registrada_path, img_referencia_path):
    """
    Avalia registro com validação de dimensões
    """
    try:
        img_reg = nib.load(img_registrada_path)
        img_ref = nib.load(img_referencia_path)

        # Verificar dimensões ANTES de processar
        if img_reg.shape != img_ref.shape:
            img_reg_data = resample_to_reference_FLIRT(img_reg, img_ref)
        else:
            img_reg_data = img_reg.get_fdata()
        
        img_ref_data = img_ref.get_fdata()

        metricas = {
            'NMI': calcular_informacao_mutua_normalizada(img_reg_data, img_ref_data),
            'NCC': calcular_correlacao_cruzada_normalizada(img_reg_data, img_ref_data),
            'MSE': calcular_erro_medio_quadratico(img_reg_data, img_ref_data),
            'MAE': calcular_erro_absoluto_medio(img_reg_data, img_ref_data)
        }

        return metricas
    
    except Exception as e:
        return {
            'NMI': np.nan,
            'NCC': np.nan,
            'MSE': np.nan,
            'MAE': np.nan,
            'erro': str(e)
        }


# TESTE T PAREADO COM COHEN'S D
def realizar_teste_t_pareado(df):
    """
    Realiza teste t pareado comparando métricas sem vs com registro.
    Remove NaN antes dos testes.
    """
    print("\n" + "="*80)
    print("TESTE T PAREADO (Paired t-test) COM TAMANHO DE EFEITO")
    print("="*80 + "\n")
    
    resultados_teste = []
    
    # Criar máscaras para dados válidos (sem NaN)
    mask_nmi = ~(df['NMI_sem_registro'].isna() | df['NMI_com_registro'].isna())
    mask_ncc = ~(df['NCC_sem_registro'].isna() | df['NCC_com_registro'].isna())
    mask_mse = ~(df['MSE_sem_registro'].isna() | df['MSE_com_registro'].isna())
    mask_mae = ~(df['MAE_sem_registro'].isna() | df['MAE_com_registro'].isna())
    
    print(f"📊 Amostras válidas por métrica:")
    print(f"   NMI: {mask_nmi.sum()}/{len(df)}")
    print(f"   NCC: {mask_ncc.sum()}/{len(df)}")
    print(f"   MSE: {mask_mse.sum()}/{len(df)}")
    print(f"   MAE: {mask_mae.sum()}/{len(df)}")
    print()
    
    # NMI (quanto maior, melhor)
    if mask_nmi.sum() > 0:
        t_nmi, p_nmi = ttest_rel(df.loc[mask_nmi, 'NMI_com_registro'], 
                                  df.loc[mask_nmi, 'NMI_sem_registro'])
        d_nmi = calcular_cohens_d(df.loc[mask_nmi, 'NMI_com_registro'], 
                                   df.loc[mask_nmi, 'NMI_sem_registro'])
    else:
        t_nmi, p_nmi, d_nmi = np.nan, np.nan, np.nan
    
    resultados_teste.append({
        'Métrica': 'NMI',
        'N_válido': mask_nmi.sum(),
        'Média_Sem': df.loc[mask_nmi, 'NMI_sem_registro'].mean(),
        'Std_Sem': df.loc[mask_nmi, 'NMI_sem_registro'].std(),
        'Média_Com': df.loc[mask_nmi, 'NMI_com_registro'].mean(),
        'Std_Com': df.loc[mask_nmi, 'NMI_com_registro'].std(),
        'Diferença_Média': df.loc[mask_nmi, 'NMI_melhoria'].mean(),
        'Estatística_t': t_nmi,
        'p-valor': p_nmi,
        "Cohen's d": d_nmi,
        'Tamanho_Efeito': interpretar_cohens_d(d_nmi),
        'Significativo (p<0.05)': 'Sim ✓' if p_nmi < 0.05 else 'Não',
        'Significativo (p<0.01)': 'Sim ✓✓' if p_nmi < 0.01 else 'Não'
    })
    
    # NCC (quanto maior, melhor)
    if mask_ncc.sum() > 0:
        t_ncc, p_ncc = ttest_rel(df.loc[mask_ncc, 'NCC_com_registro'], 
                                  df.loc[mask_ncc, 'NCC_sem_registro'])
        d_ncc = calcular_cohens_d(df.loc[mask_ncc, 'NCC_com_registro'], 
                                   df.loc[mask_ncc, 'NCC_sem_registro'])
    else:
        t_ncc, p_ncc, d_ncc = np.nan, np.nan, np.nan
    
    resultados_teste.append({
        'Métrica': 'NCC',
        'N_válido': mask_ncc.sum(),
        'Média_Sem': df.loc[mask_ncc, 'NCC_sem_registro'].mean(),
        'Std_Sem': df.loc[mask_ncc, 'NCC_sem_registro'].std(),
        'Média_Com': df.loc[mask_ncc, 'NCC_com_registro'].mean(),
        'Std_Com': df.loc[mask_ncc, 'NCC_com_registro'].std(),
        'Diferença_Média': df.loc[mask_ncc, 'NCC_melhoria'].mean(),
        'Estatística_t': t_ncc,
        'p-valor': p_ncc,
        "Cohen's d": d_ncc,
        'Tamanho_Efeito': interpretar_cohens_d(d_ncc),
        'Significativo (p<0.05)': 'Sim ✓' if p_ncc < 0.05 else 'Não',
        'Significativo (p<0.01)': 'Sim ✓✓' if p_ncc < 0.01 else 'Não'
    })
    
    # MSE (quanto menor, melhor)
    if mask_mse.sum() > 0:
        t_mse, p_mse = ttest_rel(df.loc[mask_mse, 'MSE_sem_registro'], 
                                  df.loc[mask_mse, 'MSE_com_registro'])
        d_mse = calcular_cohens_d(df.loc[mask_mse, 'MSE_sem_registro'], 
                                   df.loc[mask_mse, 'MSE_com_registro'])
    else:
        t_mse, p_mse, d_mse = np.nan, np.nan, np.nan
    
    resultados_teste.append({
        'Métrica': 'MSE',
        'N_válido': mask_mse.sum(),
        'Média_Sem': df.loc[mask_mse, 'MSE_sem_registro'].mean(),
        'Std_Sem': df.loc[mask_mse, 'MSE_sem_registro'].std(),
        'Média_Com': df.loc[mask_mse, 'MSE_com_registro'].mean(),
        'Std_Com': df.loc[mask_mse, 'MSE_com_registro'].std(),
        'Diferença_Média': df.loc[mask_mse, 'MSE_reducao'].mean(),
        'Estatística_t': t_mse,
        'p-valor': p_mse,
        "Cohen's d": d_mse,
        'Tamanho_Efeito': interpretar_cohens_d(d_mse),
        'Significativo (p<0.05)': 'Sim ✓' if not np.isnan(p_mse) and p_mse < 0.05 else 'Não',
        'Significativo (p<0.01)': 'Sim ✓✓' if not np.isnan(p_mse) and p_mse < 0.01 else 'Não'
    })
    
    # MAE (quanto menor, melhor)
    if mask_mae.sum() > 0:
        t_mae, p_mae = ttest_rel(df.loc[mask_mae, 'MAE_sem_registro'], 
                                  df.loc[mask_mae, 'MAE_com_registro'])
        d_mae = calcular_cohens_d(df.loc[mask_mae, 'MAE_sem_registro'], 
                                   df.loc[mask_mae, 'MAE_com_registro'])
    else:
        t_mae, p_mae, d_mae = np.nan, np.nan, np.nan
    
    resultados_teste.append({
        'Métrica': 'MAE',
        'N_válido': mask_mae.sum(),
        'Média_Sem': df.loc[mask_mae, 'MAE_sem_registro'].mean(),
        'Std_Sem': df.loc[mask_mae, 'MAE_sem_registro'].std(),
        'Média_Com': df.loc[mask_mae, 'MAE_com_registro'].mean(),
        'Std_Com': df.loc[mask_mae, 'MAE_com_registro'].std(),
        'Diferença_Média': df.loc[mask_mae, 'MAE_reducao'].mean(),
        'Estatística_t': t_mae,
        'p-valor': p_mae,
        "Cohen's d": d_mae,
        'Tamanho_Efeito': interpretar_cohens_d(d_mae),
        'Significativo (p<0.05)': 'Sim ✓' if not np.isnan(p_mae) and p_mae < 0.05 else 'Não',
        'Significativo (p<0.01)': 'Sim ✓✓' if not np.isnan(p_mae) and p_mae < 0.01 else 'Não'
    })
    
    df_testes = pd.DataFrame(resultados_teste)
    
    # Exibir resultados formatados
    print("Resultados do Teste T Pareado:")
    print("-" * 80)
    for _, row in df_testes.iterrows():
        print(f"\n{row['Métrica']} (n={row['N_válido']}):")
        print(f"  Sem Registro: {row['Média_Sem']:.6f} ± {row['Std_Sem']:.6f}")
        print(f"  Com Registro: {row['Média_Com']:.6f} ± {row['Std_Com']:.6f}")
        print(f"  Diferença Média: {row['Diferença_Média']:.6f}")
        
        if not np.isnan(row['Estatística_t']):
            print(f"  Estatística t: {row['Estatística_t']:.4f}")
            print(f"  p-valor: {row['p-valor']:.6f}")
            print(f"  Cohen's d: {row['Cohen\'s d']:.2f} (efeito {row['Tamanho_Efeito']})")
            print(f"  Significativo (α=0.05): {row['Significativo (p<0.05)']}")
            print(f"  Significativo (α=0.01): {row['Significativo (p<0.01)']}")
        else:
            print(f"  ⚠️ Teste não realizado (dados insuficientes)")
    
    print("\n" + "="*80)
    print("INTERPRETAÇÃO:")
    print("="*80)
    print("• p-valor < 0.05: diferença estatisticamente significativa (95% de confiança)")
    print("• p-valor < 0.01: diferença altamente significativa (99% de confiança)")
    print("• Cohen's d: |d| < 0.2 (pequeno), 0.2-0.5 (pequeno-médio), 0.5-0.8 (médio), ≥0.8 (grande)")
    print("• NMI e NCC: valores maiores indicam melhor alinhamento")
    print("• MSE e MAE: valores menores indicam melhor alinhamento")
    print("• Médias calculadas POR VOLUME (não por voxel)")
    print("="*80 + "\n")
    
    return df_testes


# COMPARAÇÃO SEM vs COM REGISTRO
def processar_comparacao_sem_vs_com_registro(base_dir, imagem_referencia):
    """Processa comparação com validação de dimensões"""
    resultados = []
    problemas = []

    dir_sem_registro = Path(base_dir) / "adni-stripped"
    dir_com_registro = Path(base_dir) / "adni-original-registered"

    if not dir_sem_registro.exists():
        print(f"⚠️  Diretório não encontrado: {dir_sem_registro}")
        return pd.DataFrame(), pd.DataFrame()

    if not dir_com_registro.exists():
        print(f"⚠️  Diretório não encontrado: {dir_com_registro}")
        return pd.DataFrame(), pd.DataFrame()

    imagens_sem = list(dir_sem_registro.glob("*.nii.gz")) + list(dir_sem_registro.glob("*.nii"))

    print(f"\n📊 Comparando {len(imagens_sem)} pares (sem registro vs com 12 DOF)...")
    print(f"🔍 Validando dimensões e integridade dos dados...\n")

    for idx, img_sem_path in enumerate(imagens_sem, 1):
        nome = img_sem_path.stem.replace(".nii", "")
        img_com_path = dir_com_registro / img_sem_path.name

        if not img_com_path.exists():
            print(f"[{idx}/{len(imagens_sem)}] {nome} ⚠️ Par não encontrado")
            problemas.append({'arquivo': nome, 'motivo': 'Par não encontrado'})
            continue

        print(f"[{idx}/{len(imagens_sem)}] {nome}...", end=" ")

        # Validar dimensões da imagem COM registro
        valida_com, info_com = validar_dimensoes(str(img_com_path))
        if not valida_com:
            print(f"⚠️ Dimensões inválidas COM registro: {info_com}")
            problemas.append({'arquivo': nome, 'motivo': f'Dimensões COM: {info_com}'})
            continue

        try:
            metricas_sem = avaliar_registro(str(img_sem_path), imagem_referencia)
            metricas_com = avaliar_registro(str(img_com_path), imagem_referencia)
            
            # Verificar se alguma métrica retornou NaN
            if any(np.isnan(v) if isinstance(v, (int, float)) else False 
                   for v in metricas_sem.values()):
                print(f"⚠️ Métricas inválidas SEM registro")
                problemas.append({'arquivo': nome, 'motivo': 'Métricas NaN SEM registro'})
                # Continua mesmo assim para incluir no relatório
            
            if any(np.isnan(v) if isinstance(v, (int, float)) else False 
                   for v in metricas_com.values()):
                print(f"⚠️ Métricas inválidas COM registro")
                problemas.append({'arquivo': nome, 'motivo': 'Métricas NaN COM registro'})

            resultado = {
                "arquivo": nome,
                "NMI_sem_registro": metricas_sem["NMI"],
                "NMI_com_registro": metricas_com["NMI"],
                "NMI_melhoria": metricas_com["NMI"] - metricas_sem["NMI"],
                "NMI_melhoria_percentual": ((metricas_com["NMI"] - metricas_sem["NMI"]) /
                                            metricas_sem["NMI"] * 100) if metricas_sem["NMI"] > 0 else np.nan,
                "NCC_sem_registro": metricas_sem["NCC"],
                "NCC_com_registro": metricas_com["NCC"],
                "NCC_melhoria": metricas_com["NCC"] - metricas_sem["NCC"],
                "MSE_sem_registro": metricas_sem["MSE"],
                "MSE_com_registro": metricas_com["MSE"],
                "MSE_reducao": metricas_sem["MSE"] - metricas_com["MSE"],
                "MAE_sem_registro": metricas_sem["MAE"],
                "MAE_com_registro": metricas_com["MAE"],
                "MAE_reducao": metricas_sem["MAE"] - metricas_com["MAE"],
            }

            resultados.append(resultado)
            print("✓")

        except Exception as e:
            print(f"✗ Erro: {e}")
            problemas.append({'arquivo': nome, 'motivo': str(e)})

    df_resultados = pd.DataFrame(resultados)
    df_problemas = pd.DataFrame(problemas)
    
    return df_resultados, df_problemas


# MAIN
def main():
    print("\n" + "="*80)
    print("AVALIAÇÃO: SEM REGISTRO vs COM REGISTRO (FLIRT 12 DOF)")
    print("COM VALIDAÇÃO E LIMPEZA DE DADOS")
    print("="*80 + "\n")

    base_dir = "./data/adni"

    imagem_ref = os.path.join(
        os.environ["FSLDIR"],
        "data/standard/MNI152_T1_2mm_brain.nii.gz"
    )

    df_completo, df_problemas = processar_comparacao_sem_vs_com_registro(base_dir, imagem_ref)

    if df_completo.empty:
        print("❌ Nenhum resultado encontrado.")
        return

    # Salvar TODOS os resultados (incluindo NaN)
    df_completo.to_csv("metricas_registro_comparacao_completo.csv", index=False)
    print(f"\n✅ Todos resultados salvos em: metricas_registro_comparacao_completo.csv")
    
    # Salvar problemas
    if not df_problemas.empty:
        df_problemas.to_csv("amostras_problematicas.csv", index=False)
        print(f"⚠️  {len(df_problemas)} amostras problemáticas salvas em: amostras_problematicas.csv")

    # Criar versão limpa (sem NaN em NENHUMA métrica)
    df_limpo = df_completo.dropna(subset=['NMI_sem_registro', 'NCC_sem_registro', 
                                           'MSE_sem_registro', 'MAE_sem_registro',
                                           'NMI_com_registro', 'NCC_com_registro',
                                           'MSE_com_registro', 'MAE_com_registro'])
    
    df_limpo.to_csv("metricas_registro_comparacao_limpo.csv", index=False)
    print(f"✅ Dados limpos ({len(df_limpo)} amostras) salvos em: metricas_registro_comparacao_limpo.csv")

    print("\n" + "="*80)
    print("RESUMO ESTATÍSTICO - DADOS LIMPOS")
    print("="*80 + "\n")
    
    print(f"📊 Total de amostras processadas: {len(df_completo)}")
    print(f"✅ Amostras válidas (sem NaN): {len(df_limpo)}")
    print(f"⚠️  Amostras com problemas: {len(df_completo) - len(df_limpo)}")

    if len(df_limpo) > 0:
        print("\n📊 SEM REGISTRO:")
        print(df_limpo[["NMI_sem_registro", "NCC_sem_registro", "MSE_sem_registro", "MAE_sem_registro"]].describe())

        print("\n📊 COM REGISTRO (12 DOF):")
        print(df_limpo[["NMI_com_registro", "NCC_com_registro", "MSE_com_registro", "MAE_com_registro"]].describe())

        print("\n📈 MELHORIAS:")
        print(df_limpo[["NMI_melhoria", "NCC_melhoria", "MSE_reducao", "MAE_reducao"]].describe())

        # REALIZAR TESTE T PAREADO COM COHEN'S D (usando dados limpos)
        df_teste_t = realizar_teste_t_pareado(df_limpo)
        
        # Salvar resultados do teste t
        df_teste_t.to_csv("teste_t_pareado_resultados.csv", index=False)
        print("✅ Resultados do teste t salvos em: teste_t_pareado_resultados.csv\n")
        
    else:
        print("\n❌ Nenhuma amostra válida encontrada para análise estatística!")
        print("   Todas as imagens têm problemas de dimensão ou dados inválidos.")
        print("   Verifique o log de amostras problemáticas.")


if __name__ == "__main__":
    main()