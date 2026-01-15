# PhysioNet Challenge 2025 - Détection de la Maladie de Chagas
## Note Complète sur les Traitements Nécessaires

**Date:** Janvier 2026
**Challenge:** Detection of Chagas Disease from the ECG
**Référence:** [PhysioNet Challenge 2025](https://moody-challenge.physionet.org/2025/)
**Papier:** [arXiv:2510.02202](https://arxiv.org/abs/2510.02202)

---

## 📋 TABLE DES MATIÈRES

1. [Contexte du Challenge](#1-contexte-du-challenge)
2. [Différences avec Challenge 2024](#2-différences-avec-challenge-2024)
3. [Architecture Nécessaire](#3-architecture-nécessaire)
4. [Composants Réutilisables](#4-composants-réutilisables)
5. [Pipeline de Traitement](#5-pipeline-de-traitement)
6. [Datasets et Préparation](#6-datasets-et-préparation)
7. [Modèles à Développer](#7-modèles-à-développer)
8. [Métriques d'Évaluation](#8-métriques-dévaluation)
9. [Roadmap de Développement](#9-roadmap-de-développement)

---

## 1. CONTEXTE DU CHALLENGE

### 🎯 Objectif
Développer des algorithmes open-source pour **identifier les cas potentiels de maladie de Chagas** à partir d'ECG 12 dérivations standard.

### 📊 Problématique Clinique
- **Maladie de Chagas** : Maladie parasitaire affectant ~6,5 millions de personnes en Amérique Centrale et du Sud
- **Mortalité** : ~10 000 décès par an
- **Problème** : Capacité de tests sérologiques limitée dans les zones endémiques
- **Solution** : Prioriser les patients pour tests de confirmation via analyse ECG

### 🏆 Critère de Victoire
- Équipe avec le **meilleur score sur le test set caché**
- Métrique : TPR (True Positive Rate) parmi les top 5% des patients classés
- **Contrainte réaliste** : 5% correspond à la capacité de test sérologique estimée au Brésil

---

## 2. DIFFÉRENCES AVEC CHALLENGE 2024

| Aspect | Challenge 2024 | Challenge 2025 |
|--------|----------------|----------------|
| **Tâche** | Numérisation d'images ECG | Classification binaire (Chagas/Non-Chagas) |
| **Input** | Images ECG imprimées (PNG) | Signaux ECG numériques (WFDB) |
| **Output** | Signaux WFDB (12 dérivations) | Probabilité Chagas [0, 1] |
| **Architecture** | Segmentation (nnU-Net) + Hough Transform | Classification (CNN/Transformer/RNN) |
| **Métrique** | SNR, ASCI, KS-distance | TPR @ top 5% |
| **Données** | PTB-XL (images synthétiques) | CODE-15%, SaMi-Trop, PTB-XL (signaux) |
| **Problème** | Vision par ordinateur | Apprentissage supervisé sur séries temporelles |

### ⚠️ Changement de Paradigme
- **2024** : Image → Signal (reconstruction)
- **2025** : Signal → Label (classification)

---

## 3. ARCHITECTURE NÉCESSAIRE

### 🏗️ Pipeline Global

```
ECG Signaux WFDB (12 dérivations, 400Hz)
    ↓
┌─────────────────────────────────────────────┐
│ 1. CHARGEMENT & VALIDATION                  │
│    - Lecture WFDB (.dat + .hea)             │
│    - Vérification intégrité (12 leads)      │
│    - Validation fréquence (400Hz)           │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 2. PRÉTRAITEMENT                             │
│    - Normalisation (z-score par lead)       │
│    - Filtrage (0.5-40Hz bandpass)           │
│    - Suppression bruit ligne base           │
│    - Détection/suppression artefacts        │
│    - Rééchantillonnage si besoin (→500Hz)   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 3. EXTRACTION FEATURES                      │
│    Option A: Features manuelles             │
│      - Intervalles QRS, QT, PR              │
│      - Variabilité HR                       │
│      - Amplitudes ondes P/Q/R/S/T           │
│      - Caractéristiques morphologiques      │
│    Option B: Features automatiques          │
│      - Embeddings CNN/Transformer           │
│      - Représentations latentes             │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 4. MODÈLE DE CLASSIFICATION                 │
│    Architecture (au choix):                 │
│    - ResNet-1D / ResNet-2D                  │
│    - Vision Transformer (ViT)              │
│    - LSTM/GRU bidirectionnel                │
│    - Ensemble (combinaison)                 │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 5. POST-TRAITEMENT                          │
│    - Calibration probabilités               │
│    - Agrégation multi-vues                  │
│    - Seuillage adaptatif                    │
└─────────────────────────────────────────────┘
    ↓
Probabilité Chagas ∈ [0, 1]
```

### 🔑 Composants Critiques

1. **Prétraitement robuste** : Gestion données bruitées/incomplètes
2. **Augmentation de données** : Essentiel pour éviter overfitting
3. **Architecture deep learning** : Capable de capturer patterns cardiaques
4. **Calibration** : Pour classement optimal (top 5%)

---

## 4. COMPOSANTS RÉUTILISABLES

### ✅ Code Actuel Directement Réutilisable

| Fichier | Utilité pour 2025 | Modifications |
|---------|-------------------|---------------|
| `src/utils/helper_code.py` | Lecture/écriture WFDB, manipulation signaux | **Aucune** - Déjà compatible |
| `config.py` | Configuration signaux (fréquence, unités, leads) | Adapter fréquence 400Hz → 500Hz |
| `requirements.txt` | Dépendances (wfdb, numpy, scipy) | Ajouter PyTorch/TensorFlow |
| `src/ptb_xl/prepare_ptbxl_data.py` | Préparation PTB-XL | Adapter pour classification |

### ⚙️ Fonctions Clés à Conserver

```python
# De helper_code.py (déjà présent)
- find_records(folder)          # Trouver fichiers WFDB
- load_header(record)            # Charger métadonnées
- load_signals(record)           # Charger signaux
- save_signals(record, signal)   # Sauvegarder résultats
- load_labels(record)            # Charger labels Chagas
```

### 🔧 Infrastructure à Adapter

- **nnU-Net** : ❌ Pas utile (segmentation ≠ classification)
- **Hough Transform** : ❌ Pas utile (travail sur signaux, pas images)
- **ecg-image-generator** : ❌ Pas utile (pas besoin de générer images)
- **Rotation detection** : ❌ Pas utile

### 💡 Philosophie à Conserver

✅ **Architecture modulaire** (séparation données/modèle/évaluation)
✅ **Scripts automatisés** (batch processing)
✅ **Validation rigoureuse** (métriques multiples)
✅ **Documentation complète**

---

## 5. PIPELINE DE TRAITEMENT

### 📂 Structure de Dossiers Proposée

```
ECG-Digitiser/
├── note 2025/                          # Cette note
├── src_2025/                           # Code Challenge 2025
│   ├── data/
│   │   ├── loader.py                   # Chargement WFDB
│   │   ├── preprocessor.py             # Prétraitement signaux
│   │   ├── augmentation.py             # Data augmentation
│   │   └── dataset.py                  # PyTorch Dataset
│   ├── models/
│   │   ├── resnet1d.py                 # ResNet pour ECG
│   │   ├── transformer.py              # Vision Transformer
│   │   ├── lstm.py                     # LSTM bidirectionnel
│   │   └── ensemble.py                 # Ensemble de modèles
│   ├── training/
│   │   ├── train.py                    # Boucle d'entraînement
│   │   ├── validate.py                 # Validation
│   │   └── losses.py                   # Fonctions de perte
│   ├── evaluation/
│   │   ├── metrics.py                  # Calcul TPR@5%
│   │   └── calibration.py              # Calibration probabilités
│   └── utils/
│       ├── helper_code.py              # Réutilisé de 2024
│       └── visualization.py            # Plots ECG + prédictions
├── data_2025/
│   ├── CODE-15%/                       # Dataset CODE-15%
│   ├── SaMi-Trop/                      # Dataset SaMi-Trop
│   ├── PTB-XL/                         # Dataset PTB-XL
│   └── processed/                      # Données prétraitées
├── models_2025/
│   ├── resnet_fold0.pth                # Checkpoints modèles
│   ├── resnet_fold1.pth
│   └── ensemble_final.pth
└── config_2025.py                      # Configuration 2025
```

### 🔄 Workflow Complet

#### **Étape 1 : Préparation des Données**

```bash
# 1.1 Télécharger les datasets
python -m src_2025.data.download_datasets

# 1.2 Convertir en format WFDB (si nécessaire)
python -m src_2025.data.convert_to_wfdb \
    -i data_2025/CODE-15%/raw \
    -o data_2025/CODE-15%/wfdb

# 1.3 Prétraiter les signaux
python -m src_2025.data.preprocess \
    -i data_2025/CODE-15%/wfdb \
    -o data_2025/processed/CODE-15% \
    --filter_bandpass 0.5 40 \
    --normalize zscore \
    --resample 500

# 1.4 Créer splits train/val/test
python -m src_2025.data.create_splits \
    -i data_2025/processed \
    -o data_2025/splits \
    --stratified \
    --k_folds 5
```

#### **Étape 2 : Entraînement**

```bash
# 2.1 Entraîner ResNet-1D (5-fold CV)
for fold in {0..4}; do
    python -m src_2025.training.train \
        --model resnet1d \
        --fold $fold \
        --epochs 100 \
        --batch_size 32 \
        --lr 0.001 \
        --device cuda:0
done

# 2.2 Entraîner Transformer
python -m src_2025.training.train \
    --model transformer \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001

# 2.3 Entraîner LSTM
python -m src_2025.training.train \
    --model lstm \
    --hidden_dim 256 \
    --num_layers 3 \
    --bidirectional
```

#### **Étape 3 : Évaluation**

```bash
# 3.1 Évaluer chaque modèle
python -m src_2025.evaluation.evaluate \
    --model models_2025/resnet_fold0.pth \
    --data data_2025/processed/test \
    --metric tpr_at_5pct

# 3.2 Créer ensemble
python -m src_2025.models.ensemble \
    --models models_2025/resnet_*.pth models_2025/transformer.pth \
    --weights 0.4 0.4 0.4 0.4 0.4 0.6 \
    --output models_2025/ensemble_final.pth

# 3.3 Calibration
python -m src_2025.evaluation.calibration \
    --model models_2025/ensemble_final.pth \
    --data data_2025/processed/val \
    --method isotonic
```

#### **Étape 4 : Prédiction & Soumission**

```bash
# 4.1 Prédire sur test set
python -m src_2025.predict \
    --model models_2025/ensemble_final.pth \
    --data test_folder \
    --output predictions.csv

# 4.2 Créer fichier de soumission
python -m src_2025.create_submission \
    --predictions predictions.csv \
    --output submission_2025.zip
```

---

## 6. DATASETS ET PRÉPARATION

### 📊 Datasets Disponibles

#### **CODE-15% Dataset**
- **Source** : Brésil (2010-2016)
- **Taille** : >300 000 enregistrements ECG 12 dérivations
- **Durée** : 7.3s ou 10.2s
- **Fréquence** : 400 Hz
- **Labels** : Auto-rapportés (non validés sérologiquement)
- **⚠️ Attention** : Labels potentiellement bruités

#### **SaMi-Trop Dataset**
- **Source** : Brésil
- **Taille** : 1631 ECG de 1959 patients
- **Validation** : ✅ Sérologiquement confirmée
- **Durée** : ~10s
- **Fréquence** : 400 Hz
- **Qualité** : ⭐ Dataset de référence (gold standard)

#### **PTB-XL Dataset**
- **Source** : Allemagne
- **Taille** : 21 837 enregistrements
- **Durée** : 10s
- **Fréquence** : 500 Hz (aussi 100Hz disponible)
- **Labels** : Diagnostics variés (pas Chagas)
- **Usage** : Pré-entraînement / Transfert learning

### 🔄 Prétraitement Nécessaire

#### **1. Nettoyage des Signaux**

```python
# Filtrage passe-bande (0.5-40 Hz)
def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=400, order=4):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

# Suppression dérive baseline (filtre médian)
def remove_baseline_wander(signal, window_size=200):
    from scipy.signal import medfilt
    baseline = medfilt(signal, kernel_size=window_size)
    return signal - baseline

# Normalisation par dérivation
def normalize_per_lead(signal):
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std
```

#### **2. Gestion des Données Manquantes**

```python
# Vérifier intégrité
def check_signal_integrity(signal, expected_leads=12):
    if signal.shape[1] != expected_leads:
        raise ValueError(f"Expected {expected_leads} leads, got {signal.shape[1]}")

    # Vérifier NaN
    if np.isnan(signal).any():
        # Interpolation linéaire
        from scipy.interpolate import interp1d
        for i in range(signal.shape[1]):
            mask = ~np.isnan(signal[:, i])
            if mask.sum() > 0:
                f = interp1d(np.where(mask)[0], signal[mask, i],
                            kind='linear', fill_value='extrapolate')
                signal[:, i] = f(np.arange(len(signal)))

    return signal
```

#### **3. Harmonisation Fréquences**

```python
# Rééchantillonnage vers fréquence cible
def resample_signal(signal, fs_original, fs_target=500):
    from scipy.signal import resample
    num_samples_new = int(len(signal) * fs_target / fs_original)
    return resample(signal, num_samples_new, axis=0)

# Exemple: CODE-15% (400Hz) → 500Hz
signal_resampled = resample_signal(signal, fs_original=400, fs_target=500)
```

#### **4. Augmentation de Données**

```python
# Techniques d'augmentation
class ECGAugmentation:
    def __init__(self):
        pass

    def time_warp(self, signal, sigma=0.2):
        """Déformation temporelle"""
        from scipy.ndimage import map_coordinates
        time_steps = np.arange(len(signal))
        warp = np.random.normal(0, sigma, len(signal)).cumsum()
        warped_time = time_steps + warp
        warped_time = np.clip(warped_time, 0, len(signal)-1)
        return map_coordinates(signal, [warped_time], order=1, mode='nearest')

    def amplitude_scale(self, signal, scale_range=(0.9, 1.1)):
        """Mise à l'échelle amplitude"""
        scale = np.random.uniform(*scale_range)
        return signal * scale

    def add_gaussian_noise(self, signal, noise_level=0.01):
        """Ajout bruit gaussien"""
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise

    def time_shift(self, signal, shift_range=50):
        """Décalage temporel"""
        shift = np.random.randint(-shift_range, shift_range)
        return np.roll(signal, shift, axis=0)

    def lead_dropout(self, signal, dropout_prob=0.1):
        """Masquage aléatoire dérivations"""
        mask = np.random.random(signal.shape[1]) > dropout_prob
        signal_aug = signal.copy()
        signal_aug[:, ~mask] = 0
        return signal_aug
```

---

## 7. MODÈLES À DÉVELOPPER

### 🧠 Architecture 1 : ResNet-1D

```python
import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D_ECG(nn.Module):
    def __init__(self, num_leads=12, num_classes=1):
        super().__init__()

        # Input: (batch, 12 leads, time_steps)
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResBlock1D(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 12, time_steps)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)  # Probabilité [0, 1]
        return x
```

### 🌟 Architecture 2 : Vision Transformer pour ECG

```python
class ECGViT(nn.Module):
    def __init__(self,
                 num_leads=12,
                 seq_length=5000,  # 10s @ 500Hz
                 patch_size=50,    # Patch de 0.1s
                 embed_dim=256,
                 num_heads=8,
                 num_layers=6,
                 num_classes=1):
        super().__init__()

        self.num_patches = seq_length // patch_size

        # Patch embedding
        self.patch_embed = nn.Conv1d(num_leads, embed_dim,
                                      kernel_size=patch_size,
                                      stride=patch_size)

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, 12, seq_length)
        x = self.patch_embed(x)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)    # (batch, num_patches, embed_dim)
        x = x + self.pos_embed

        x = self.transformer(x)

        # Global average pooling over patches
        x = x.mean(dim=1)

        x = self.classifier(x)
        return x
```

### 🔁 Architecture 3 : LSTM Bidirectionnel

```python
class BiLSTM_ECG(nn.Module):
    def __init__(self,
                 num_leads=12,
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.3,
                 num_classes=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_leads,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_length, 12)
        lstm_out, _ = self.lstm(x)  # (batch, seq_length, hidden_dim*2)

        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # (batch, seq_length, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_dim*2)

        out = self.classifier(context)
        return out
```

### 🎯 Architecture 4 : Ensemble

```python
class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)

    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        # Weighted average
        stacked = torch.stack(predictions, dim=0)
        weighted = stacked * self.weights.view(-1, 1, 1)
        ensemble_pred = weighted.sum(dim=0)

        return ensemble_pred
```

---

## 8. MÉTRIQUES D'ÉVALUATION

### 📈 Métrique Principale : TPR @ Top 5%

```python
def compute_tpr_at_top_k_percent(y_true, y_pred_proba, k=5):
    """
    Calcule le True Positive Rate parmi les top k% des prédictions.

    Args:
        y_true: Labels réels (0 ou 1)
        y_pred_proba: Probabilités prédites [0, 1]
        k: Pourcentage (5 pour top 5%)

    Returns:
        TPR @ top k%
    """
    n = len(y_true)
    n_top_k = max(1, int(n * k / 100))

    # Trier par probabilité décroissante
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    top_k_indices = sorted_indices[:n_top_k]

    # Calculer TPR
    y_true_top_k = y_true[top_k_indices]
    n_positives_total = y_true.sum()
    n_positives_top_k = y_true_top_k.sum()

    if n_positives_total == 0:
        return 0.0

    tpr = n_positives_top_k / n_positives_total
    return tpr

# Exemple d'utilisation
y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
y_pred = np.array([0.1, 0.9, 0.3, 0.7, 0.8, 0.2, 0.4, 0.6])

tpr_5 = compute_tpr_at_top_k_percent(y_true, y_pred, k=5)
print(f"TPR @ Top 5%: {tpr_5:.3f}")
```

### 📊 Métriques Secondaires (pour analyse)

```python
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix
)

def compute_all_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        'tpr_at_5pct': compute_tpr_at_top_k_percent(y_true, y_pred_proba, k=5),
        'auroc': roc_auc_score(y_true, y_pred_proba),
        'auprc': average_precision_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    return metrics
```

### 🎯 Fonction de Perte Adaptée

```python
class RankingLoss(nn.Module):
    """Loss qui optimise directement le ranking (TPR@5%)"""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        # Composante 1: Binary Cross-Entropy classique
        bce_loss = self.bce(y_pred, y_true)

        # Composante 2: Pairwise ranking loss
        # Pénalise quand positif a score < négatif
        pos_mask = (y_true == 1).squeeze()
        neg_mask = (y_true == 0).squeeze()

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = y_pred[pos_mask]
            neg_scores = y_pred[neg_mask]

            # Toutes les paires (pos, neg)
            pos_expanded = pos_scores.unsqueeze(1)  # (n_pos, 1)
            neg_expanded = neg_scores.unsqueeze(0)  # (1, n_neg)

            # max(0, margin + neg_score - pos_score)
            margin = 0.1
            ranking_loss = torch.clamp(margin + neg_expanded - pos_expanded, min=0).mean()
        else:
            ranking_loss = 0

        # Combinaison
        total_loss = (1 - self.alpha) * bce_loss + self.alpha * ranking_loss
        return total_loss
```

---

## 9. ROADMAP DE DÉVELOPPEMENT

### 🗓️ Phase 1 : Infrastructure (Semaines 1-2)

- [ ] Créer structure de dossiers `src_2025/`
- [ ] Télécharger datasets (CODE-15%, SaMi-Trop, PTB-XL)
- [ ] Adapter `helper_code.py` pour chargement Chagas labels
- [ ] Implémenter prétraitement de base (filtrage, normalisation)
- [ ] Créer DataLoader PyTorch
- [ ] Mettre en place pipeline d'augmentation
- [ ] Configurer logging et tracking (Weights & Biases / MLflow)

### 🗓️ Phase 2 : Baseline (Semaines 3-4)

- [ ] Implémenter ResNet-1D simple
- [ ] Entraîner sur SaMi-Trop (dataset de qualité)
- [ ] Valider métrique TPR@5%
- [ ] Analyser erreurs et patterns
- [ ] Tester sur CODE-15%
- [ ] Établir baseline performance

### 🗓️ Phase 3 : Modèles Avancés (Semaines 5-7)

- [ ] Implémenter Vision Transformer
- [ ] Implémenter BiLSTM avec attention
- [ ] Pré-entraînement sur PTB-XL
- [ ] Fine-tuning sur données Chagas
- [ ] Hyperparameter tuning (learning rate, architecture depth)
- [ ] Cross-validation 5-fold

### 🗓️ Phase 4 : Ensemble & Optimisation (Semaines 8-9)

- [ ] Créer modèle ensemble (ResNet + ViT + LSTM)
- [ ] Optimiser poids d'ensemble
- [ ] Calibration probabilités (Platt scaling, Isotonic)
- [ ] Post-processing avancé
- [ ] Test-time augmentation (TTA)

### 🗓️ Phase 5 : Validation & Soumission (Semaine 10)

- [ ] Validation finale sur hold-out set
- [ ] Analyse d'erreurs détaillée
- [ ] Génération visualisations
- [ ] Préparation code soumission
- [ ] Documentation complète
- [ ] Soumission officielle

### 🗓️ Phase 6 : Itérations (Selon résultats)

- [ ] Analyse feedback du leaderboard
- [ ] Identification weaknesses
- [ ] Nouvelles features / architectures
- [ ] Re-soumission

---

## 10. CHECKLIST CRITIQUE

### ✅ Avant de Commencer

- [ ] Lire attentivement [règlement officiel](https://moody-challenge.physionet.org/2025/)
- [ ] Télécharger code exemple Python : [physionetchallenges/python-example-2025](https://github.com/physionetchallenges/python-example-2025)
- [ ] Comprendre format de soumission
- [ ] Vérifier dates limites
- [ ] S'inscrire sur la plateforme

### ✅ Qualité des Données

- [ ] Vérifier distribution classes (déséquilibre ?)
- [ ] Analyser qualité signaux (bruit, artefacts)
- [ ] Identifier données manquantes/invalides
- [ ] Stratification train/val/test
- [ ] Cohérence fréquences échantillonnage

### ✅ Entraînement

- [ ] Monitoring overfitting (early stopping)
- [ ] Stratégie learning rate (scheduler)
- [ ] Régularisation (dropout, weight decay)
- [ ] Gestion déséquilibre classes (class weights, focal loss)
- [ ] Reproductibilité (seed fixe)

### ✅ Évaluation

- [ ] Validation croisée stratifiée
- [ ] Calcul correct métrique TPR@5%
- [ ] Courbe ROC / Precision-Recall
- [ ] Analyse par sous-groupes (âge, sexe, etc.)
- [ ] Calibration plot

### ✅ Soumission

- [ ] Code conforme aux spécifications
- [ ] Dockerfile fonctionnel
- [ ] Temps d'inférence respecté
- [ ] Format output correct
- [ ] Documentation complète

---

## 11. RESSOURCES UTILES

### 📚 Papers à Lire

1. **Challenge 2025 Paper** : [arXiv:2510.02202](https://arxiv.org/abs/2510.02202)
2. **Chagas Disease & ECG** :
   - "ECG features of Chagas disease" (multiple reviews)
   - Right bundle branch block (RBBB) patterns
3. **Deep Learning for ECG** :
   - "Cardiologist-level arrhythmia detection" (Rajpurkar et al., 2017)
   - "Deep learning for ECG analysis" (Hong et al., 2020)

### 🔗 Liens Importants

- [Challenge officiel 2025](https://moody-challenge.physionet.org/2025/)
- [Code exemple Python](https://github.com/physionetchallenges/python-example-2025)
- [Code évaluation](https://github.com/physionetchallenges/evaluation-2025)
- [Forum discussions](https://groups.google.com/g/physionet-challenges)
- [PTB-XL dataset](https://physionet.org/content/ptb-xl/)

### 🛠️ Outils & Librairies

```python
# Traitement signaux
wfdb          # Lecture/écriture WFDB
scipy         # Filtrage, traitement signal
neurokit2     # Extraction features ECG

# Deep Learning
torch         # PyTorch
transformers  # Hugging Face (pour ViT)
timm          # PyTorch Image Models

# Évaluation
scikit-learn  # Métriques ML
wandb         # Experiment tracking

# Visualisation
matplotlib
seaborn
plotly
```

---

## 12. NOTES FINALES

### ⚠️ Pièges à Éviter

1. **Overfitting sur SaMi-Trop** : Dataset petit (1631 ECG)
2. **Labels bruités CODE-15%** : Auto-rapportés, non validés
3. **Déséquilibre classes** : Chagas = minorité
4. **Différences distributions** : Train ≠ Test (distributional shift)
5. **Optimiser mauvaise métrique** : Accuracy ≠ TPR@5%

### 💡 Astuces pour Gagner

1. **Ensemble diversifié** : Combiner architectures très différentes
2. **Pré-entraînement** : Utiliser PTB-XL (21K ECG)
3. **Data cleaning** : Filtrer CODE-15% (qualité variable)
4. **Calibration** : Essentielle pour ranking optimal
5. **Domain knowledge** : Incorporer features ECG connues (QRS, QT)
6. **Test-time augmentation** : Moyenner prédictions sur augmentations

### 🎯 Objectif Réaliste

- **Baseline** : TPR@5% ~ 0.30-0.40
- **Bon modèle** : TPR@5% ~ 0.50-0.60
- **Top 3** : TPR@5% > 0.65

---

**Bon courage pour le Challenge 2025 !** 🚀

---

_Note créée le : Janvier 2026_
_Dernière mise à jour : Janvier 2026_
