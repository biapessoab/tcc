# Análise do Impacto do Registro de Imagens no Desempenho de Classificadores Por Redes Neurais Profundas

**Trabalho de Conclusão de Curso - Ciência da Computação**  
Pontifícia Universidade Católica de Minas Gerais (PUC Minas)

**Aluna:** Ana Beatriz Pessoa Braz  
**Orientador:** Alexei Manso Correa Machado

## Sobre o Projeto

Este trabalho investiga o problema computacional de registro de volumes, analisando como essa etapa de pré-processamento influencia o aprendizado e o desempenho de modelos de classificação profunda aplicados a dados tridimensionais. O estudo foca na avaliação do impacto do alinhamento geométrico sobre redes neurais, utilizando o conjunto de dados ADNI e um modelo ResNet-10 3D para realizar a classificação binária entre pacientes com Alzheimer e controles normais.

## Estrutura do Repositório

```
tcc/
├── data/              # Dados da base ADNI
├── scripts/              # Scripts Python para processamento
├── ADNIMERGE.csv        # Dados clínicos e demográficos do ADNI
├── MNI152_T1_2mm_brain.nii.gz  # Template de referência MNI152
└── readme.md            
```

## Objetivo

Investigar como o registro volumétrico aplicado como etapa de pré-processamento afeta o desempenho de classificadores baseados em redes neurais profundas, comparando resultados com e sem registro e analisando separadamente os efeitos de cada tipo de transformação geométrica (translação, rotação e escala).

## Pipeline de Processamento

### Etapas Comuns
1. **Download dos Dados**: Volumes de ressonância magnética do banco ADNI no formato NIfTI
2. **Filtragem por Resolução**: Seleção de volumes com resolução consistente
3. **Organização por Diagnóstico**: Classificação binária entre DEMENTIA e CN (controle normal)
4. **Reorientação**: Ajuste para o padrão anatômico MNI152 usando `fslreorient2std`
5. **Skull Stripping**: Remoção de tecidos extracerebrais com BET (Brain Extraction Tool)

### Etapas de Registro (quando aplicável)
6. **Registro Linear**: Alinhamento ao atlas MNI152 usando FLIRT (FMRIB's Linear Image Registration Tool)
   - Registro afim completo (translação + rotação + escala + cisalhamento)
   - Transformações isoladas (apenas translação, apenas rotação, apenas escala)

### Classificação
7. **Modelo ResNet-10 3D**: Arquitetura com skip connections para classificação binária
8. **Validação Cruzada**: 10 folds com divisão por pacientes para evitar data leakage

## Como Usar

### Instalação

```bash
git clone https://github.com/biapessoab/tcc.git
cd tcc
```

### Execução

Siga os passos do pipeline sequencialmente executando os scripts na pasta `scripts/`.

## Dados

- **ADNIMERGE.csv**: Arquivo contendo informações clínicas, demográficas e de diagnóstico dos participantes do estudo ADNI.
- **MNI152_T1_2mm_brain.nii.gz**: Template de referência do espaço estereotáxico MNI152 utilizado para registro espacial.

## 👤 Autor

Ana Beatriz Pessoa Braz - [@biapessoab](https://github.com/biapessoab)

---

**Nota**: Este projeto utiliza dados do ADNI. O uso desses dados deve estar em conformidade com os termos de uso do ADNI.