import numpy as np
from scipy import stats
import pandas as pd

# DADOS DOS EXPERIMENTOS (10-FOLD CV)

# AUC por fold 
auc_data = {
    'SEM_REGISTRO': [0.8210, 0.7434, 0.8006, 0.6546, 0.5860, 0.8316, 0.4479, 0.6663, 0.6937, 0.6833],
    'REGISTRO_COMPLETO': [0.9181, 0.9002, 0.9062, 0.6916, 0.6799, 0.7947, 0.8071, 0.7951, 0.7844, 0.8764],
    'APENAS_TRANSLACAO': [0.6295, 0.5000, 0.7625, 0.5100, 0.6362, 0.4274, 0.5721, 0.5598, 0.6948, 0.8472],
    'APENAS_ROTACAO': [0.8295, 0.6250, 0.6266, 0.5605, 0.7593, 0.6411, 0.5621, 0.6755, 0.7146, 0.7208],
    'APENAS_ESCALA': [0.4057, 0.8114, 0.7077, 0.8291, 0.7434, 0.6326, 0.4457, 0.7931, 0.6625, 0.5653],
}

# F1-Score (Macro) por fold
f1_data = {
    'SEM_REGISTRO': [0.3500, 0.2209, 0.5928, 0.3243, 0.3293, 0.4185, 0.2588, 0.5714, 0.3404, 0.5349],
    'REGISTRO_COMPLETO': [0.5113, 0.7214, 0.3402, 0.3182, 0.6362, 0.8058, 0.6791, 0.4457, 0.3404, 0.8009],
    'APENAS_TRANSLACAO': [0.3158, 0.4174, 0.3402, 0.3421, 0.4589, 0.4579, 0.3043, 0.3152, 0.3600, 0.5227],
    'APENAS_ROTACAO': [0.7538, 0.2439, 0.3671, 0.4533, 0.6842, 0.3762, 0.3942, 0.5644, 0.3929, 0.2632],
    'APENAS_ESCALA': [0.3500, 0.4711, 0.5272, 0.5273, 0.3293, 0.6204, 0.3942, 0.5758, 0.3404, 0.5204],
}

# Precision (Macro) por fold
precision_data = {
    'SEM_REGISTRO': [0.2692, 0.1418, 0.7074, 0.2400, 0.2500, 0.8065, 0.1746, 0.6462, 0.2581, 0.8396],
    'REGISTRO_COMPLETO': [0.7917, 0.7166, 0.2578, 0.2365, 0.6362, 0.8444, 0.8148, 0.6547, 0.2581, 0.8099],
    'APENAS_TRANSLACAO': [0.2308, 0.3582, 0.2578, 0.2600, 0.6600, 0.8115, 0.4167, 0.2302, 0.7459, 0.7115],
    'APENAS_ROTACAO': [0.7610, 0.6439, 0.5081, 0.5812, 0.7142, 0.3016, 0.3254, 0.5952, 0.4911, 0.1786],
    'APENAS_ESCALA': [0.2692, 0.8636, 0.6685, 0.7868, 0.2455, 0.6529, 0.3254, 0.6931, 0.2581, 0.5256],
}

# Recall (Macro) por fold
recall_data = {
    'SEM_REGISTRO': [0.5000, 0.5000, 0.6344, 0.5000, 0.4821, 0.5200, 0.5000, 0.5963, 0.5000, 0.5750],
    'REGISTRO_COMPLETO': [0.5833, 0.7593, 0.5000, 0.4861, 0.6362, 0.7937, 0.6696, 0.5370, 0.5000, 0.7944],
    'APENAS_TRANSLACAO': [0.5000, 0.5000, 0.5000, 0.5000, 0.5529, 0.5400, 0.4684, 0.5000, 0.5156, 0.5611],
    'APENAS_ROTACAO': [0.7595, 0.5104, 0.5010, 0.5310, 0.6938, 0.5000, 0.5000, 0.5852, 0.4969, 0.5000],
    'APENAS_ESCALA': [0.5000, 0.5263, 0.5890, 0.5972, 0.5000, 0.6211, 0.5000, 0.6273, 0.5000, 0.5278],
}

# Accuracy por fold
accuracy_data = {
    'SEM_REGISTRO': [0.5385, 0.2836, 0.6250, 0.4800, 0.4909, 0.6190, 0.3492, 0.6190, 0.5161, 0.6964],
    'REGISTRO_COMPLETO': [0.6154, 0.7463, 0.5156, 0.4667, 0.6364, 0.8254, 0.7619, 0.5714, 0.5161, 0.8214],
    'APENAS_TRANSLACAO': [0.4615, 0.7164, 0.5156, 0.5200, 0.5455, 0.6349, 0.3492, 0.4603, 0.5000, 0.6786],
    'APENAS_ROTACAO': [0.7538, 0.2985, 0.5156, 0.5467, 0.6909, 0.6032, 0.6508, 0.5714, 0.4839, 0.3571],
    'APENAS_ESCALA': [0.5385, 0.7313, 0.5781, 0.6133, 0.4909, 0.6667, 0.6508, 0.6032, 0.5161, 0.5357],
}

# CRITÉRIO DE CONVERGÊNCIA
# True = Completou 15 épocas (CONVERGIU)
# False = Parou antes (NÃO CONVERGIU - early stopping)

convergencia = {
    'SEM_REGISTRO': [False, False, True, True, False, False, False, False, True, True],
    'REGISTRO_COMPLETO': [True, True, True, True, True, False, True, False, False, True],
    'APENAS_TRANSLACAO': [False, False, False, True, False, False, False, False, False, False],
    'APENAS_ROTACAO': [False, False, False, False, True, False, False, False, False, False],
    'APENAS_ESCALA': [False, True, False, True, False, True, False, True, False, False],
}

def cohen_d_paired(x1, x2):
    """Calcula Cohen's d para dados pareados"""
    diff = np.array(x1) - np.array(x2)
    return np.mean(diff) / np.std(diff, ddof=1)

def criar_tabela_comparacao(exp1_name, exp2_name, metricas_dict, convergencia_dict, apenas_convergentes=False):
    """Cria tabela com estatísticas comparativas entre dois experimentos"""
    resultados = []
    
    # Identificar folds válidos
    if apenas_convergentes:
        conv1 = convergencia_dict[exp1_name]
        conv2 = convergencia_dict[exp2_name]
        folds_validos = [i for i in range(10) if conv1[i] and conv2[i]]
        n_pares = len(folds_validos)
        tipo_analise = f"Convergentes (n={n_pares})"
    else:
        folds_validos = list(range(10))
        n_pares = 10
        tipo_analise = f"Todos (n={n_pares})"
    
    if n_pares < 2:
        return None, None
    
    for metrica_nome, metrica_data in metricas_dict.items():
        # Extrair valores dos folds válidos
        vals1 = [metrica_data[exp1_name][i] for i in folds_validos]
        vals2 = [metrica_data[exp2_name][i] for i in folds_validos]
        
        # Estatísticas descritivas
        media1 = np.mean(vals1)
        dp1 = np.std(vals1, ddof=1)
        media2 = np.mean(vals2)
        dp2 = np.std(vals2, ddof=1)
        
        # Teste t pareado
        t_stat, p_valor = stats.ttest_rel(vals1, vals2)
        
        # Tamanho do efeito (Cohen's d)
        d = cohen_d_paired(vals1, vals2)
        
        resultados.append({
            'Métrica': metrica_nome,
            f'{exp1_name} (média±DP)': f'{media1:.3f}±{dp1:.3f}',
            f'{exp2_name} (média±DP)': f'{media2:.3f}±{dp2:.3f}',
            't': f'{t_stat:.3f}',
            'p': f'{p_valor:.4f}',
            "Cohen's d": f'{d:.3f}'
        })
    
    return pd.DataFrame(resultados), tipo_analise

def mostrar_convergencia():
    """Mostra estatísticas de convergência por experimento"""
    print("\n📊 ESTATÍSTICAS DE CONVERGÊNCIA (Completaram 15 épocas):")
    print("-"*80)
    for exp_name, conv_list in convergencia.items():
        n_conv = sum(conv_list)
        taxa = (n_conv / 10) * 100
        folds_conv = [i+1 for i, c in enumerate(conv_list) if c]
        print(f"{exp_name:25s}: {n_conv}/10 ({taxa:5.1f}%) - Folds: {folds_conv}")

# HIPÓTESE 1: MELHORIA NA DISCRIMINAÇÃO (APENAS CONVERGENTES)
print("HIPÓTESE 1: MELHORIA NA DISCRIMINAÇÃO ENTRE CLASSES")

mostrar_convergencia()

metricas_dict = {
    'AUC': auc_data,
    'F1-Score': f1_data,
    'Precision': precision_data,
    'Recall': recall_data,
    'Accuracy': accuracy_data,
}

# Comparação principal: REGISTRO vs SEM REGISTRO
print("\n" + "-"*80)
print("Comparação: REGISTRO_COMPLETO vs SEM_REGISTRO")
print("-"*80)
df_h1_principal, tipo_h1 = criar_tabela_comparacao(
    'REGISTRO_COMPLETO', 'SEM_REGISTRO',
    metricas_dict, convergencia,
    apenas_convergentes=True
)

if df_h1_principal is not None:
    print(f"\n{tipo_h1} pares onde AMBOS convergiram\n")
    print(df_h1_principal.to_string(index=False))
else:
    print("\n⚠️  POUCOS PARES CONVERGENTES - Análise estatística inviável")

# Comparações adicionais
comparacoes_h1 = [
    ('REGISTRO_COMPLETO', 'APENAS_TRANSLACAO'),
    ('REGISTRO_COMPLETO', 'APENAS_ROTACAO'),
    ('REGISTRO_COMPLETO', 'APENAS_ESCALA'),
    ('APENAS_TRANSLACAO', 'SEM_REGISTRO'),
    ('APENAS_ROTACAO', 'SEM_REGISTRO'),
    ('APENAS_ESCALA', 'SEM_REGISTRO'),
]

for exp1, exp2 in comparacoes_h1:
    print(f"\n{'-'*80}")
    print(f"Comparação: {exp1} vs {exp2}")
    print("-"*80)
    df_comp, tipo = criar_tabela_comparacao(
        exp1, exp2,
        metricas_dict, convergencia,
        apenas_convergentes=True
    )
    if df_comp is not None:
        print(f"\n{tipo}\n")
        print(df_comp.to_string(index=False))
    else:
        print("\n⚠️  POUCOS PARES CONVERGENTES - Análise estatística inviável")

# HIPÓTESE 2: ESTABILIDADE DO APRENDIZADO (TODOS OS FOLDS)
print("HIPÓTESE 2: ROBUSTEZ E GENERALIZAÇÃO DO APRENDIZADO")

# Comparação principal: REGISTRO vs SEM REGISTRO
print("\n" + "-"*80)
print("Comparação: REGISTRO_COMPLETO vs SEM_REGISTRO")
print("-"*80)
df_h2_principal, tipo_h2 = criar_tabela_comparacao(
    'REGISTRO_COMPLETO', 'SEM_REGISTRO',
    metricas_dict, convergencia,
    apenas_convergentes=False
)
print(f"\n{tipo_h2}\n")
print(df_h2_principal.to_string(index=False))

# Comparações com transformações isoladas
comparacoes_h2 = [
    ('REGISTRO_COMPLETO', 'APENAS_TRANSLACAO'),
    ('REGISTRO_COMPLETO', 'APENAS_ROTACAO'),
    ('REGISTRO_COMPLETO', 'APENAS_ESCALA'),
    ('APENAS_TRANSLACAO', 'SEM_REGISTRO'),
    ('APENAS_ROTACAO', 'SEM_REGISTRO'),
    ('APENAS_ESCALA', 'SEM_REGISTRO'),
]

for exp1, exp2 in comparacoes_h2:
    print(f"\n{'-'*80}")
    print(f"Comparação: {exp1} vs {exp2}")
    print("-"*80)
    df_comp, tipo = criar_tabela_comparacao(
        exp1, exp2,
        metricas_dict, convergencia,
        apenas_convergentes=False
    )
    print(f"\n{tipo}\n")
    print(df_comp.to_string(index=False))

# RESUMO 
print("\n\n" + "="*80)
print("RESUMO")
print("="*80)

mostrar_convergencia()

print("\n\n📊 MÉDIAS GERAIS (Todos os 10 folds):\n")
resumo = []
for exp_name in auc_data.keys():
    resumo.append({
        'Experimento': exp_name,
        'AUC': f"{np.mean(auc_data[exp_name]):.3f}±{np.std(auc_data[exp_name], ddof=1):.3f}",
        'F1': f"{np.mean(f1_data[exp_name]):.3f}±{np.std(f1_data[exp_name], ddof=1):.3f}",
        'Accuracy': f"{np.mean(accuracy_data[exp_name]):.3f}±{np.std(accuracy_data[exp_name], ddof=1):.3f}",
    })

df_resumo = pd.DataFrame(resumo)
print(df_resumo.to_string(index=False))