# Améliorations Proposées pour le PhysioNet Challenge 2025

## 📋 Vue d'Ensemble

Ce document liste toutes les améliorations recommandées pour maximiser les chances de victoire au PhysioNet Challenge 2025. Les améliorations sont classées par priorité et impact attendu.

---

## 🎯 PRIORITÉ HAUTE (Impact Majeur)

### 1. Architecture Multi-Échelle

**Problème actuel** : Le repo 2024 utilise une seule échelle de traitement (nnU-Net 2D).

**Amélioration** :
- Créer architecture qui capture patterns à différentes échelles temporelles
- Combiner CNN locaux (morphologie ondes) + Transformers globaux (rythme)
- Implémenter pyramide multi-résolution

```python
class MultiScaleECG(nn.Module):
    def __init__(self):
        super().__init__()
        # Échelle 1: Morphologie fine (haute fréquence)
        self.local_cnn = ResNet1D(kernel_sizes=[3, 5, 7])

        # Échelle 2: Patterns moyens (complexes QRS)
        self.mid_cnn = ResNet1D(kernel_sizes=[15, 31, 63])

        # Échelle 3: Rythme global (basse fréquence)
        self.global_transformer = ECGViT(patch_size=100)

        # Fusion
        self.fusion = nn.Linear(512*3, 1)

    def forward(self, x):
        local_feat = self.local_cnn(x)
        mid_feat = self.mid_cnn(x)
        global_feat = self.global_transformer(x)

        combined = torch.cat([local_feat, mid_feat, global_feat], dim=1)
        return self.fusion(combined)
```

**Impact attendu** : +5-10% TPR@5%

---

### 2. Pré-entraînement sur PTB-XL + Transfer Learning

**Problème actuel** : Pas de pré-entraînement, entraînement from scratch.

**Amélioration** :
- **Phase 1** : Pré-entraîner sur PTB-XL (21K ECG) avec tâche auxiliaire (classification diagnostics)
- **Phase 2** : Fine-tuning sur SaMi-Trop (labels Chagas validés)
- **Phase 3** : Fine-tuning final sur CODE-15% (large dataset)

```python
# Étape 1: Pré-entraînement multi-tâches
class PretrainModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Prédire 5 diagnostics PTB-XL
        self.head_ptbxl = nn.Linear(512, 5)
        # Prédire 12 dérivations séparément (tâche auxiliaire)
        self.head_leads = nn.Linear(512, 12)

    def forward(self, x):
        features = self.backbone(x)
        diag = self.head_ptbxl(features)
        leads = self.head_leads(features)
        return diag, leads

# Étape 2: Fine-tuning Chagas
model.head_chagas = nn.Linear(512, 1)  # Remplacer tête
# Geler couches basses, fine-tuner couches hautes
for param in model.backbone.layer1.parameters():
    param.requires_grad = False
```

**Impact attendu** : +8-15% TPR@5%

---

### 3. Stratégie d'Ensemble Avancée

**Problème actuel** : Pas d'ensemble dans le repo 2024 (un seul modèle nnU-Net).

**Amélioration** :
- Ensemble hétérogène (ResNet + ViT + LSTM + XGBoost sur features)
- Stacking avec meta-learner (apprend à combiner prédictions)
- Blending optimal via Bayesian optimization

```python
# Ensemble via stacking
class StackingEnsemble(nn.Module):
    def __init__(self, base_models):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)

        # Meta-learner (prend prédictions base models en input)
        self.meta_learner = nn.Sequential(
            nn.Linear(len(base_models), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Prédictions base models
        base_preds = []
        for model in self.base_models:
            with torch.no_grad():  # Pas de backprop à travers base models
                pred = model(x)
                base_preds.append(pred)

        base_preds = torch.cat(base_preds, dim=1)

        # Meta-learner combine intelligemment
        final_pred = self.meta_learner(base_preds)
        return final_pred

# Entraînement 2 phases:
# 1. Entraîner base models indépendamment
# 2. Entraîner meta-learner sur validation set
```

**Impact attendu** : +5-8% TPR@5%

---

### 4. Loss Function Optimisée pour Ranking

**Problème actuel** : BCELoss standard (optimise accuracy, pas TPR@5%).

**Amélioration** :
- Loss qui maximise directement TPR@5%
- Pénaliser fortement cas positifs mal classés en bas du ranking

```python
class TPRLoss(nn.Module):
    """Loss qui optimise TPR@top-k%"""

    def __init__(self, k=5, lambda_bce=0.3):
        super().__init__()
        self.k = k
        self.lambda_bce = lambda_bce
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        n_top_k = max(1, int(batch_size * self.k / 100))

        # Composante 1: BCE standard
        bce_loss = self.bce(y_pred, y_true)

        # Composante 2: Pénaliser positifs hors top-k
        sorted_indices = torch.argsort(y_pred.squeeze(), descending=True)
        top_k_mask = torch.zeros_like(y_true, dtype=torch.bool)
        top_k_mask[sorted_indices[:n_top_k]] = True

        # Positifs qui ne sont PAS dans top-k : forte pénalité
        missed_positives = (y_true == 1) & (~top_k_mask)
        penalty = (1 - y_pred[missed_positives]).sum()

        # Composante 3: Récompenser positifs dans top-k
        caught_positives = (y_true == 1) & (top_k_mask)
        reward = -y_pred[caught_positives].sum()  # Négatif = récompense

        total_loss = self.lambda_bce * bce_loss + penalty + reward
        return total_loss
```

**Impact attendu** : +3-7% TPR@5%

---

### 5. Data Augmentation ECG-Spécifique

**Problème actuel** : Le repo 2024 fait de l'augmentation d'images (rotation, brightness), pas pertinent pour signaux.

**Amélioration** :
- Augmentations physiquement plausibles (pas juste bruit aléatoire)
- Mixup adapté aux ECG

```python
class SmartECGAugmentation:
    def __init__(self):
        pass

    def heart_rate_variation(self, signal, hr_factor_range=(0.9, 1.1)):
        """Simule variation fréquence cardiaque"""
        factor = np.random.uniform(*hr_factor_range)
        from scipy.signal import resample
        new_len = int(len(signal) * factor)
        resampled = resample(signal, new_len, axis=0)

        # Recadrer/padder à longueur originale
        if len(resampled) > len(signal):
            start = (len(resampled) - len(signal)) // 2
            return resampled[start:start+len(signal)]
        else:
            pad = (len(signal) - len(resampled)) // 2
            return np.pad(resampled, ((pad, len(signal)-len(resampled)-pad), (0, 0)))

    def baseline_wander(self, signal, amplitude=0.05):
        """Ajoute dérive baseline réaliste"""
        from scipy.signal import butter, filtfilt
        # Basse fréquence (0.1-0.5 Hz)
        b, a = butter(2, [0.1, 0.5], btype='band', fs=500)
        noise = np.random.randn(len(signal), signal.shape[1])
        wander = filtfilt(b, a, noise, axis=0) * amplitude
        return signal + wander

    def powerline_interference(self, signal, freq=60, amplitude=0.01):
        """Ajoute interférence ligne électrique (50/60Hz)"""
        t = np.arange(len(signal)) / 500  # Temps
        interference = amplitude * np.sin(2 * np.pi * freq * t)[:, None]
        return signal + interference

    def mixup_ecg(self, signal1, label1, signal2, label2, alpha=0.4):
        """Mixup adapté aux ECG (mélange signaux)"""
        lam = np.random.beta(alpha, alpha)

        # Aligner phases (détection pics R)
        from scipy.signal import find_peaks
        peaks1 = find_peaks(signal1[:, 1], distance=200)[0]  # Lead II
        peaks2 = find_peaks(signal2[:, 1], distance=200)[0]

        # Shift signal2 pour aligner premiers pics
        if len(peaks1) > 0 and len(peaks2) > 0:
            shift = peaks1[0] - peaks2[0]
            signal2 = np.roll(signal2, shift, axis=0)

        mixed_signal = lam * signal1 + (1 - lam) * signal2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_signal, mixed_label

    def lead_swap(self, signal, prob=0.2):
        """Échange dérivations similaires (ex: V1↔V2)"""
        if np.random.random() < prob:
            # Paires échangeables: (V1,V2), (V3,V4), (V5,V6), (aVR,aVL)
            pairs = [(6, 7), (8, 9), (10, 11), (3, 4)]  # Indices leads
            pair = pairs[np.random.randint(len(pairs))]
            signal[:, [pair[0], pair[1]]] = signal[:, [pair[1], pair[0]]]
        return signal
```

**Impact attendu** : +4-6% TPR@5%

---

## 🎯 PRIORITÉ MOYENNE (Impact Modéré)

### 6. Feature Engineering Clinique

**Amélioration** :
- Extraire features ECG connues pour Chagas (RBBB, fascicular blocks)
- Combiner avec features deep learning

```python
def extract_clinical_features(signal, fs=500):
    """Extrait features cliniques pertinentes pour Chagas"""
    import neurokit2 as nk

    features = {}

    # Détection ondes
    signals, info = nk.ecg_process(signal[:, 1], sampling_rate=fs)  # Lead II

    # Intervalles
    features['pr_interval'] = np.mean(info['ECG_P_Offsets'] - info['ECG_P_Onsets']) / fs
    features['qrs_duration'] = np.mean(info['ECG_R_Offsets'] - info['ECG_R_Onsets']) / fs
    features['qt_interval'] = np.mean(info['ECG_T_Offsets'] - info['ECG_R_Peaks']) / fs

    # Right Bundle Branch Block (RBBB) - caractéristique Chagas
    qrs_v1 = signal[:, 6].max() - signal[:, 6].min()  # Amplitude V1
    features['rbbb_score'] = qrs_v1 * features['qrs_duration']  # Score RBBB

    # Variabilité fréquence cardiaque
    rr_intervals = np.diff(info['ECG_R_Peaks']) / fs
    features['hrv_rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))

    # Caractéristiques morphologiques
    features['r_amplitude'] = np.mean([signal[:, i].max() for i in range(12)])
    features['t_amplitude'] = np.mean([signals['ECG_T_Peaks'].mean()])

    return features

# Combiner avec deep learning
class HybridModel(nn.Module):
    def __init__(self, cnn_backbone):
        super().__init__()
        self.cnn = cnn_backbone  # Features automatiques
        self.clinical_mlp = nn.Sequential(  # Features cliniques
            nn.Linear(10, 32),  # 10 features cliniques
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        # Fusion
        self.classifier = nn.Linear(512 + 32, 1)

    def forward(self, x, clinical_features):
        cnn_feat = self.cnn(x)
        clinical_feat = self.clinical_mlp(clinical_features)
        combined = torch.cat([cnn_feat, clinical_feat], dim=1)
        return self.classifier(combined)
```

**Impact attendu** : +3-5% TPR@5%

---

### 7. Calibration Avancée des Probabilités

**Amélioration** :
- Température scaling
- Isotonic regression
- Beta calibration

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class ProbabilityCalibrator:
    def __init__(self, method='isotonic'):
        self.method = method
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == 'platt':
            self.calibrator = LogisticRegression()

    def fit(self, y_pred, y_true):
        if self.method == 'isotonic':
            self.calibrator.fit(y_pred, y_true)
        elif self.method == 'platt':
            self.calibrator.fit(y_pred.reshape(-1, 1), y_true)

    def transform(self, y_pred):
        if self.method == 'isotonic':
            return self.calibrator.predict(y_pred)
        elif self.method == 'platt':
            return self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]

# Utilisation
calibrator = ProbabilityCalibrator(method='isotonic')
calibrator.fit(val_predictions, val_labels)
calibrated_test_pred = calibrator.transform(test_predictions)
```

**Impact attendu** : +2-4% TPR@5%

---

### 8. Gestion Intelligente du Déséquilibre de Classes

**Amélioration** :
- Focal Loss (pénalise cas faciles)
- Class-balanced sampling
- SMOTE pour signaux

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        bce = -y_true * torch.log(y_pred + 1e-7) - (1 - y_true) * torch.log(1 - y_pred + 1e-7)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        loss = alpha_weight * focal_weight * bce
        return loss.mean()

# Sampler équilibré
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler
```

**Impact attendu** : +2-3% TPR@5%

---

### 9. Test-Time Augmentation (TTA)

**Amélioration** :
- Moyenner prédictions sur plusieurs augmentations au test

```python
def predict_with_tta(model, signal, num_augmentations=10):
    """Prédiction avec TTA"""
    augmenter = SmartECGAugmentation()
    predictions = []

    # Prédiction originale
    with torch.no_grad():
        pred_orig = model(torch.tensor(signal).unsqueeze(0)).item()
    predictions.append(pred_orig)

    # Prédictions sur versions augmentées
    for _ in range(num_augmentations):
        aug_signal = signal.copy()

        # Appliquer augmentations légères
        aug_signal = augmenter.baseline_wander(aug_signal, amplitude=0.02)
        aug_signal = augmenter.add_gaussian_noise(aug_signal, noise_level=0.01)
        aug_signal = augmenter.time_shift(aug_signal, shift_range=20)

        with torch.no_grad():
            pred = model(torch.tensor(aug_signal).unsqueeze(0)).item()
        predictions.append(pred)

    # Moyenne
    final_pred = np.mean(predictions)
    return final_pred
```

**Impact attendu** : +1-3% TPR@5%

---

## 🎯 PRIORITÉ BASSE (Optimisations)

### 10. Attention Mécanismes

**Amélioration** :
- Attention sur dérivations (certaines plus informatives)
- Attention temporelle (focus sur zones importantes)

```python
class LeadAttention(nn.Module):
    """Attention sur les 12 dérivations"""
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lead_features):
        # lead_features: (batch, 12, hidden_dim)
        attn_weights = self.attention(lead_features)  # (batch, 12, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Pondérer features par attention
        weighted = lead_features * attn_weights
        aggregated = weighted.sum(dim=1)  # (batch, hidden_dim)

        return aggregated, attn_weights  # Retourner poids pour interprétabilité
```

**Impact attendu** : +1-2% TPR@5%

---

### 11. Knowledge Distillation

**Amélioration** :
- Entraîner modèle léger (student) à imiter ensemble lourd (teacher)

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    """
    Loss pour knowledge distillation
    alpha=0: pure distillation
    alpha=1: pure supervision
    """
    # Soft targets (teacher)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_predictions = F.log_softmax(student_logits / temperature, dim=1)
    distill_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)

    # Hard targets (labels)
    ce_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

    return alpha * ce_loss + (1 - alpha) * distill_loss
```

**Impact attendu** : +1-2% TPR@5% (modèle plus rapide)

---

### 12. Explainability (Grad-CAM pour ECG)

**Amélioration** :
- Visualiser zones ECG importantes pour prédiction
- Validation clinique

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_signal, class_idx=0):
        # Forward
        output = self.model(input_signal)

        # Backward
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Calcul CAM
        weights = self.gradients.mean(dim=2, keepdim=True)  # Global average pooling
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # Normaliser
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.detach().cpu().numpy()

# Utilisation
import matplotlib.pyplot as plt

gradcam = GradCAM(model, target_layer=model.layer4)
cam = gradcam.generate_cam(input_signal)

# Visualiser
plt.figure(figsize=(15, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)
    plt.plot(input_signal[0, i].cpu().numpy(), alpha=0.6, label=f'Lead {i+1}')
    plt.plot(cam[0], 'r', alpha=0.5, label='Importance')
    plt.legend()
plt.tight_layout()
plt.savefig('gradcam_ecg.png')
```

**Impact attendu** : +0% TPR@5% (mais crucial pour validation clinique et confiance)

---

## 🔄 AMÉLIORATIONS INFRASTRUCTURE

### 13. Pipeline MLOps Complet

**Amélioration** :
- Tracking expériences (Weights & Biases)
- Versioning données (DVC)
- CI/CD pour entraînement
- Monitoring performances

```python
import wandb

# Initialisation
wandb.init(project='physionet-2025', name='resnet1d-fold0')

# Logging pendant training
wandb.config.update({
    'architecture': 'ResNet1D',
    'learning_rate': 0.001,
    'batch_size': 32,
    'augmentation': 'smart_ecg'
})

# Log métriques
for epoch in range(num_epochs):
    train_loss, train_tpr = train_epoch(...)
    val_loss, val_tpr = validate(...)

    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_tpr@5': train_tpr,
        'val_loss': val_loss,
        'val_tpr@5': val_tpr
    })

    # Log modèle
    if val_tpr > best_tpr:
        torch.save(model.state_dict(), 'best_model.pth')
        wandb.save('best_model.pth')
```

---

### 14. Hyperparameter Tuning Automatisé

**Amélioration** :
- Optuna pour recherche hyperparamètres
- Bayesian optimization

```python
import optuna

def objective(trial):
    # Hyperparamètres à optimiser
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512, step=64)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    # Créer et entraîner modèle
    model = ResNet1D(hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Entraînement rapide (quelques epochs)
    for epoch in range(20):
        train(model, optimizer, batch_size=batch_size)

    # Évaluer
    val_tpr = evaluate(model)

    return val_tpr  # Optuna maximise cette métrique

# Lancer optimisation
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f'Meilleurs hyperparamètres: {study.best_params}')
```

---

### 15. Data Pipeline Efficace

**Amélioration** :
- Prétraitement en batch (multiprocessing)
- Caching signaux prétraités
- DataLoader optimisé

```python
import multiprocessing as mp
from functools import partial

def preprocess_single_record(record_path, output_dir):
    """Prétraite un enregistrement"""
    signal, fields = load_signals(record_path)

    # Prétraitement
    signal = bandpass_filter(signal)
    signal = remove_baseline_wander(signal)
    signal = normalize_per_lead(signal)

    # Sauvegarder
    output_path = os.path.join(output_dir, f'{os.path.basename(record_path)}.npy')
    np.save(output_path, signal)

def preprocess_batch(record_paths, output_dir, num_workers=12):
    """Prétraite batch en parallèle"""
    os.makedirs(output_dir, exist_ok=True)

    with mp.Pool(num_workers) as pool:
        func = partial(preprocess_single_record, output_dir=output_dir)
        pool.map(func, record_paths)

# Utilisation
record_paths = find_records('data_2025/CODE-15%/')
preprocess_batch(record_paths, 'data_2025/processed/', num_workers=12)
```

---

## 📊 TABLEAU RÉCAPITULATIF

| Amélioration | Priorité | Impact TPR@5% | Difficulté | Temps Estimé |
|--------------|----------|---------------|------------|--------------|
| Architecture Multi-Échelle | Haute | +5-10% | Moyenne | 1-2 semaines |
| Pré-entraînement PTB-XL | Haute | +8-15% | Moyenne | 1-2 semaines |
| Ensemble Avancé | Haute | +5-8% | Moyenne | 1 semaine |
| Loss Optimisée Ranking | Haute | +3-7% | Faible | 3-5 jours |
| Data Augmentation ECG | Haute | +4-6% | Faible | 3-5 jours |
| Feature Engineering Clinique | Moyenne | +3-5% | Moyenne | 1 semaine |
| Calibration Avancée | Moyenne | +2-4% | Faible | 2-3 jours |
| Gestion Déséquilibre | Moyenne | +2-3% | Faible | 2-3 jours |
| Test-Time Augmentation | Moyenne | +1-3% | Faible | 1-2 jours |
| Attention Mécanismes | Basse | +1-2% | Moyenne | 3-5 jours |
| Knowledge Distillation | Basse | +1-2% | Moyenne | 3-5 jours |
| Explainability (Grad-CAM) | Basse | 0% | Faible | 2-3 jours |
| MLOps Pipeline | Infra | N/A | Moyenne | 3-5 jours |
| Hyperparameter Tuning | Infra | +2-5% | Faible | 1-2 jours |
| Data Pipeline Efficace | Infra | N/A | Faible | 1-2 jours |

---

## 🎯 STRATÉGIE RECOMMANDÉE

### Phase 1 (Semaines 1-3) - Fondations Solides
1. ✅ Pré-entraînement PTB-XL
2. ✅ Data Augmentation ECG
3. ✅ Loss Optimisée Ranking
4. ✅ Gestion Déséquilibre Classes

**Résultat attendu** : Baseline TPR@5% ~ 0.50

---

### Phase 2 (Semaines 4-6) - Performance Boost
1. ✅ Architecture Multi-Échelle
2. ✅ Feature Engineering Clinique
3. ✅ Calibration Avancée

**Résultat attendu** : TPR@5% ~ 0.60-0.65

---

### Phase 3 (Semaines 7-9) - Excellence
1. ✅ Ensemble Avancé (3-5 modèles)
2. ✅ Test-Time Augmentation
3. ✅ Hyperparameter Tuning Final

**Résultat attendu** : TPR@5% ~ 0.68-0.72

---

### Phase 4 (Semaine 10) - Polish & Validation
1. ✅ Explainability (validation clinique)
2. ✅ Code cleanup
3. ✅ Documentation
4. ✅ Soumission finale

**Résultat attendu** : Top 3 du Challenge 🏆

---

## 💡 CONSEILS GÉNÉRAUX

### Do's ✅
- **Itérer rapidement** : Tester idées vite sur petit subset
- **Valider rigoureusement** : 5-fold CV stratifiée obligatoire
- **Analyser erreurs** : Comprendre cas où modèle échoue
- **Documenter tout** : Expériences, résultats, intuitions
- **Collaborer** : Si équipe, diviser travail efficacement

### Don'ts ❌
- **Pas d'overfitting** : Attention SaMi-Trop (petit dataset)
- **Pas d'optimisation prématurée** : Focus d'abord sur métrique principale
- **Pas de complexité inutile** : Simple > Complexe si même perf
- **Pas de data leakage** : Validation stricte train/val/test
- **Pas de tout ré-entraîner** : Réutiliser checkpoints intermédiaires

---

## 📚 RESSOURCES SUPPLÉMENTAIRES

### Papers à Implémenter

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Vision Transformer pour ECG

2. **"Focal Loss for Dense Object Detection"** (Lin et al., 2017)
   - Gérer déséquilibre classes

3. **"mixup: Beyond Empirical Risk Minimization"** (Zhang et al., 2018)
   - Augmentation par mélange

4. **"Learning to Rank for Information Retrieval"** (Liu, 2009)
   - Loss optimisée pour ranking

5. **"Do We Need Hundreds of Classifiers to Solve Real World Classification Problems?"** (Fernández-Delgado et al., 2014)
   - Importance ensembles

---

## 🔚 CONCLUSION

**Gain Total Estimé** : +25-35% TPR@5% par rapport à baseline simple

**Timeline Réaliste** : 10 semaines de développement intensif

**Risque Technique** : Faible (toutes améliorations bien documentées)

**Probabilité Top 3** : Élevée (> 70%) si implémentation rigoureuse

---

**Bonne chance ! 🚀 N'hésitez pas à adapter ces améliorations selon vos résultats expérimentaux.**

---

_Document créé : Janvier 2026_
_Dernière mise à jour : Janvier 2026_
