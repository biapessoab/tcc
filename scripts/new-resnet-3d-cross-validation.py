import os
import glob
import numpy as np
import nibabel as nib
from skimage.transform import resize
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import resnet
import random
import re 
import time

# === CONFIGURAÇÕES GERAIS ===
RES_SIZE = (91, 109, 91)
TEST_SIZE = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 4
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 5  # Early stopping
N_FOLDS = 5

# === Caminhos ===
REG_PATHS = [
    '/Users/anabeatrizbraz/Documents/PUC/tcc/data/adni/adni-registered/*CN_stripped_affine.nii.gz',
    '/Users/anabeatrizbraz/Documents/PUC/tcc/data/adni/adni-registered/*Dementia_stripped_affine.nii.gz'
]

UNREG_PATHS = [
    '/Users/anabeatrizbraz/Documents/PUC/tcc/data/adni/adni-stripped/*CN_stripped.nii.gz',
    '/Users/anabeatrizbraz/Documents/PUC/tcc/data/adni/adni-stripped/*Dementia_stripped.nii.gz'
]

# === Seed Global ===
def set_seed(seed):
    """Define seed para todas as bibliotecas"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# === Funções Auxiliares ===
def extract_id_and_label(filename):
    """Extrai ID e label do arquivo ADNI"""
    base = os.path.basename(filename)
    match = re.search(r'(ADNI_\d{3}_S_\d{4})', base)
    if not match:
        return None, None
    subject_id = match.group(1)
    
    if "-CN" in base:
        label = 0
    elif "Dementia" in base:
        label = 1
    else:
        return None, None
    
    return subject_id, label

def load_and_preprocess_volume(f):
    """Carrega e preprocessa volume NIfTI"""
    try:
        img = nib.load(f)
        a = img.get_fdata()
    except Exception:
        return None
    
    if a.ndim > 3:
        a = a[..., 0]
    
    a = np.nan_to_num(a)
    
    # Normalização [0,1]
    m, mi = np.max(a), np.min(a)
    if m - mi > 0:
        a = (a - mi) / (m - mi + 1e-8)
    
    # Z-score
    a = (a - a.mean()) / (a.std() + 1e-8)
    
    # Resize
    a = resize(a, RES_SIZE, anti_aliasing=True)
    a = np.expand_dims(a, axis=0)
    
    return a.astype(np.float32)

# === Dataset PyTorch ===
class BrainDataset(Dataset):
    def __init__(self, file_list, ids_filter=None):
        self.samples = []
        for f in file_list:
            subject_id, label = extract_id_and_label(f)
            if subject_id is not None and label is not None:
                if ids_filter is None or subject_id in ids_filter:
                    vol = load_and_preprocess_volume(f)
                    if vol is not None:
                        self.samples.append((vol, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# === Preparação dos Dados ===
def get_all_files_and_ids(paths):
    """Coleta arquivos e IDs únicos"""
    all_files = []
    for path in paths:
        all_files.extend(glob.glob(path))
    
    id_label_pairs = [extract_id_and_label(f) for f in all_files]
    valid_ids = [id for id, label in id_label_pairs if id is not None]
    unique_ids = np.unique(valid_ids)
    
    return all_files, unique_ids

# === Modelo e Treino ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(seed):
    """Cria modelo com seed específica"""
    set_seed(seed)
    model = resnet.resnet10(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2
    ).to(device)
    return model

def train_model_with_validation(model, train_loader, val_loader, exp_name, 
                                 max_epochs=MAX_EPOCHS, lr=LEARNING_RATE, patience=PATIENCE):
    """
    Treina modelo COM EARLY STOPPING baseado em validação
    """
    print(f"\n🏁 Treinando {exp_name}...")
    
    # Pesos de classe
    train_labels = [label for _, label in train_loader.dataset.samples]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor([1.0 / count if count > 0 else 1.0 for count in class_counts], 
                                  dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * 2
    
    print(f"Pesos de Classe: {class_weights.cpu().numpy().round(3)}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                             factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # --- TREINO ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping na época {epoch+1}")
                break
    
    # Restaura melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, test_loader):
    """Avalia modelo e retorna métricas"""
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs[:,1].cpu().numpy())
    
    # Métricas
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, 
                                                                 average='weighted', zero_division=0)
    
    return {
        'auc': roc_auc,
        'cm': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true,
        'y_probs': y_probs
    }

# === Cross-Validation ===
def cross_validate_fair(files, description, all_ids):
    """10-Fold CV e early stopping"""
    
    fold_results = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    total_start = time.time()
    
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_ids), 1):
        print(f"\n{'='*60}")
        print(f"=== {description} - Fold {fold}/{N_FOLDS} ===")
        print(f"{'='*60}")
        fold_start = time.time()
        
        # Split train/val/test
        train_val_ids = all_ids[train_val_idx]
        test_ids = all_ids[test_idx]
        
        # Divide train_val em train e val (80/20)
        train_ids, val_ids = train_val_ids[:int(0.8*len(train_val_ids))], \
                              train_val_ids[int(0.8*len(train_val_ids)):]
        
        # Datasets
        train_dataset = BrainDataset(files, ids_filter=train_ids)
        val_dataset = BrainDataset(files, ids_filter=val_ids)
        test_dataset = BrainDataset(files, ids_filter=test_ids)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Modelo com seed específica por fold
        fold_seed = RANDOM_SEED + fold
        model = create_model(fold_seed)
        
        # Treino com validação
        model = train_model_with_validation(model, train_loader, val_loader, 
                                             f"{description} - Fold {fold}")
        
        # Avaliação no test set
        results = evaluate_model(model, test_loader)
        fold_results.append(results)
        
        fold_end = time.time()
        
        print(f"\n📊 Resultados Fold {fold}:")
        print(f"AUC: {results['auc']:.4f}")
        print(f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F1: {results['f1']:.4f}")
        print(f"Matriz de Confusão:\n{results['cm']}")
        print(f"⏱️ Tempo do Fold: {fold_end - fold_start:.2f}s")
    
    total_end = time.time()
    
    # Estatísticas
    print(f"\n{'='*60}")
    print(f"=== {description} - RESULTADOS FINAIS (5-Fold CV) ===")
    print(f"{'='*60}")
    
    aucs = [r['auc'] for r in fold_results]
    precisions = [r['precision'] for r in fold_results]
    recalls = [r['recall'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    cms = [r['cm'] for r in fold_results]
    
    print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Matriz de Confusão Média:\n{np.mean(cms, axis=0).astype(int)}")
    print(f"⏱️ Tempo Total: {total_end - total_start:.2f}s")
    
    return fold_results

# === EXECUÇÃO PRINCIPAL ===
if __name__ == '__main__':
    reg_files, reg_unique_ids = get_all_files_and_ids(REG_PATHS)
    unreg_files, _ = get_all_files_and_ids(UNREG_PATHS)
    
    print(f"Total de sujeitos únicos: {len(reg_unique_ids)}")
    print(f"Total de volumes REGISTRADOS: {len(reg_files)}")
    print(f"Total de volumes NÃO REGISTRADOS: {len(unreg_files)}")
    
    all_ids = np.array(reg_unique_ids)
    
    reg_results = cross_validate_fair(reg_files, "REGISTRADO", all_ids)
    unreg_results = cross_validate_fair(unreg_files, "NÃO REGISTRADO", all_ids)
    
    print(f"\n{'='*60}")
    print("=== COMPARAÇÃO FINAL ===")
    print(f"{'='*60}")
    
    reg_aucs = [r['auc'] for r in reg_results]
    unreg_aucs = [r['auc'] for r in unreg_results]
    
    print(f"REGISTRADO:     AUC = {np.mean(reg_aucs):.4f} ± {np.std(reg_aucs):.4f}")
    print(f"NÃO REGISTRADO: AUC = {np.mean(unreg_aucs):.4f} ± {np.std(unreg_aucs):.4f}")
    print(f"Diferença: {(np.mean(reg_aucs) - np.mean(unreg_aucs))*100:.2f}%")