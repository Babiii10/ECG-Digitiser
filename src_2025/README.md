# PhysioNet Challenge 2025 - Scripts d'Entraînement et Prédiction

## 📋 Vue d'Ensemble

Ce répertoire contient tous les scripts Python nécessaires pour participer au **PhysioNet Challenge 2025** (Détection de la maladie de Chagas à partir d'ECG).

### Structure des Fichiers

```
src_2025/
├── data/
│   ├── preprocessing.py      # Prétraitement des signaux ECG
│   └── dataset.py            # Dataset PyTorch avec augmentation
├── models/
│   ├── resnet1d.py          # Architectures ResNet-1D
│   └── losses.py            # Fonctions de loss customisées
├── training/
│   ├── train.py             # Script d'entraînement complet
│   └── predict.py           # Script de prédiction
└── utils/
    └── metrics.py           # Métriques d'évaluation (TPR@5%)
```

## 🚀 Utilisation Rapide

### 1. Entraînement du Modèle

```bash
python -m src_2025.training.train \
    --data_folder data_2025/SaMi-Trop \
    --output_dir models_2025/resnet_medium \
    --model_type resnet1d_medium \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --loss_type focal
```

**Arguments principaux:**
- `--data_folder`: Dossier contenant les données WFDB
- `--output_dir`: Dossier de sortie pour les checkpoints
- `--model_type`: Architecture (`resnet1d_small`, `resnet1d_medium`, `resnet1d_large`, `seresnet1d`)
- `--epochs`: Nombre d'epochs
- `--batch_size`: Taille des batchs
- `--learning_rate`: Taux d'apprentissage
- `--loss_type`: Type de loss (`bce`, `focal`, `ranking`, `tpr`)

### 2. Prédiction

```bash
python -m src_2025.training.predict \
    --model_path models_2025/resnet_medium/model_best.pth \
    --data_folder data_2025/test \
    --output_file predictions.csv \
    --use_tta
```

**Arguments principaux:**
- `--model_path`: Chemin vers le checkpoint du modèle
- `--data_folder`: Dossier contenant les données de test
- `--output_file`: Fichier CSV de sortie avec les prédictions
- `--use_tta`: Activer Test-Time Augmentation (améliore les performances)

### 3. Interface Challenge (team_code.py)

```bash
# Entraînement via l'interface challenge
python train_model.py -d training_data -m model

# Prédiction via l'interface challenge
python run_model.py -d test_data -m model -o predictions
```

## 📊 Pipeline Complet de Prétraitement

Le prétraitement est automatique et inclut:

1. **Filtrage passe-bande (0.5-40 Hz)** - Supprime bruit et dérive baseline
2. **Filtre notch (50/60 Hz)** - Supprime interférence ligne électrique
3. **Suppression dérive baseline** - Filtre médian
4. **Détection et suppression artefacts** - Seuillage z-score + interpolation
5. **Rééchantillonnage (→500 Hz)** - Harmonisation fréquence
6. **Normalisation par dérivation** - Z-score normalization
7. **Padding/Truncation** - Longueur fixe (5000 samples = 10s @ 500Hz)

### Exemple de Prétraitement Manuel

```python
from src_2025.data.preprocessing import ECGPreprocessor

# Créer preprocessor
preprocessor = ECGPreprocessor(
    target_fs=500,
    lowcut=0.5,
    highcut=40,
    notch_freq=60
)

# Prétraiter un signal
processed = preprocessor.preprocess(
    ecg_signal,           # (n_samples, 12)
    fs_original=400,
    target_length=5000
)
```

## 🔧 Augmentation de Données

L'augmentation est appliquée automatiquement pendant l'entraînement:

- **Time warping** - Déformation temporelle
- **Amplitude scaling** - Mise à l'échelle amplitude
- **Gaussian noise** - Bruit gaussien
- **Time shift** - Décalage temporel
- **Lead dropout** - Masquage aléatoire dérivations
- **Baseline wander** - Dérive baseline réaliste

### Exemple d'Augmentation Manuelle

```python
from src_2025.data.preprocessing import ECGAugmentation

# Créer augmenter
augmenter = ECGAugmentation(
    time_warp_sigma=0.2,
    amplitude_scale_range=(0.9, 1.1),
    noise_level=0.01
)

# Augmenter un signal
augmented = augmenter.augment(signal, prob=0.5)
```

## 🧠 Architectures Disponibles

### 1. ResNet-1D Small (Rapide)
```python
from src_2025.models.resnet1d import create_resnet1d_small
model = create_resnet1d_small()  # ~500K paramètres
```

### 2. ResNet-1D Medium (Recommandé)
```python
from src_2025.models.resnet1d import create_resnet1d_medium
model = create_resnet1d_medium()  # ~2M paramètres
```

### 3. ResNet-1D Large (Haute Performance)
```python
from src_2025.models.resnet1d import create_resnet1d_large
model = create_resnet1d_large()  # ~10M paramètres
```

### 4. SE-ResNet-1D (Avec Attention)
```python
from src_2025.models.resnet1d import create_seresnet1d
model = create_seresnet1d()  # ~2.5M paramètres
```

## 📉 Fonctions de Loss

### 1. Binary Cross-Entropy (Baseline)
```bash
--loss_type bce
```

### 2. Focal Loss (Gestion Déséquilibre)
```bash
--loss_type focal
```
Recommandé pour données déséquilibrées.

### 3. Ranking Loss (Optimisé TPR@5%)
```bash
--loss_type ranking
```
Pénalise positifs classés sous négatifs.

### 4. TPR Loss (Optimisé Directement TPR@5%)
```bash
--loss_type tpr
```
Maximise directement la métrique du challenge.

### 5. Combined Loss (Ensemble)
```bash
--loss_type combined
```
Combine Focal + Ranking avec poids apprenables.

## 📊 Métriques d'Évaluation

### Métrique Principale: TPR @ Top 5%

```python
from src_2025.utils.metrics import compute_tpr_at_top_k_percent

tpr_5 = compute_tpr_at_top_k_percent(y_true, y_pred_proba, k=5)
print(f"TPR @ Top 5%: {tpr_5:.4f}")
```

### Métriques Secondaires

```python
from src_2025.utils.metrics import compute_all_metrics

metrics = compute_all_metrics(y_true, y_pred_proba, threshold=0.5)

print(f"AUROC: {metrics['auroc']:.4f}")
print(f"AUPRC: {metrics['auprc']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
```

## 🎯 Exemples d'Utilisation Avancée

### 1. Entraînement Multi-GPU

```bash
python -m src_2025.training.train \
    --data_folder data_2025/CODE-15% \
    --output_dir models_2025/multi_gpu \
    --device cuda \
    --batch_size 64 \
    --num_workers 8
```

### 2. Fine-tuning depuis Checkpoint

```python
# Dans train.py, charger checkpoint pré-entraîné
checkpoint = torch.load('models_2025/pretrained/model_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Geler couches basses
for param in model.layer1.parameters():
    param.requires_grad = False
```

### 3. Ensemble de Modèles

```python
# Charger plusieurs modèles
model1 = torch.load('model1.pth')
model2 = torch.load('model2.pth')
model3 = torch.load('model3.pth')

# Prédiction ensemble (moyenne)
with torch.no_grad():
    pred1 = torch.sigmoid(model1(x))
    pred2 = torch.sigmoid(model2(x))
    pred3 = torch.sigmoid(model3(x))

    ensemble_pred = (pred1 + pred2 + pred3) / 3
```

### 4. Test-Time Augmentation

```bash
# Activer TTA (améliore TPR@5% de 1-3%)
python -m src_2025.training.predict \
    --model_path models_2025/resnet_medium/model_best.pth \
    --data_folder data_2025/test \
    --output_file predictions_tta.csv \
    --use_tta \
    --n_augmentations 10
```

## ⚙️ Configuration Optimale (Recommandée)

```bash
python -m src_2025.training.train \
    --data_folder data_2025/SaMi-Trop \
    --output_dir models_2025/optimal \
    --model_type resnet1d_medium \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --weight_decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --loss_type focal \
    --grad_clip 1.0 \
    --train_ratio 0.8 \
    --target_length 5000 \
    --target_fs 500 \
    --device cuda \
    --seed 42
```

## 🔍 Monitoring et Logging

Les métriques sont sauvegardées automatiquement:

```
models_2025/resnet_medium/
├── config.json              # Configuration utilisée
├── history.json             # Historique train/val
├── checkpoint_latest.pth    # Dernier checkpoint
├── checkpoint_best.pth      # Meilleur checkpoint
└── model_best.pth           # Meilleurs poids seulement
```

### Visualiser l'Historique

```python
import json
import matplotlib.pyplot as plt

# Charger historique
with open('models_2025/resnet_medium/history.json', 'r') as f:
    history = json.load(f)

# Plot TPR@5%
train_tpr = [m['tpr_at_5pct'] for m in history['train']]
val_tpr = [m['tpr_at_5pct'] for m in history['val']]

plt.plot(train_tpr, label='Train')
plt.plot(val_tpr, label='Val')
plt.xlabel('Epoch')
plt.ylabel('TPR @ Top 5%')
plt.legend()
plt.savefig('tpr_history.png')
```

## 🐛 Debugging

### Vérifier Prétraitement

```python
from src_2025.data.preprocessing import ECGPreprocessor
import matplotlib.pyplot as plt

preprocessor = ECGPreprocessor()

# Charger signal
signal, fields = helper_code.load_signals('record_path')

# Prétraiter
processed = preprocessor.preprocess(signal, fs_original=400)

# Visualiser
plt.figure(figsize=(15, 6))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.plot(processed[:, i])
    plt.title(f'Lead {i+1}')
plt.tight_layout()
plt.savefig('preprocessed_signal.png')
```

### Tester Dataset

```python
from src_2025.data.dataset import ChagasECGDataset

dataset = ChagasECGDataset(
    data_folder='data_2025/SaMi-Trop',
    augment=False
)

print(f"Dataset size: {len(dataset)}")

# Tester premier sample
signal, label, demographics = dataset[0]
print(f"Signal shape: {signal.shape}")
print(f"Label: {label}")
print(f"Demographics: {demographics}")
```

## 📝 Notes Importantes

1. **Mémoire GPU**: Ajuster `--batch_size` selon votre GPU
   - 8 GB VRAM: batch_size=16
   - 12 GB VRAM: batch_size=32
   - 24 GB VRAM: batch_size=64

2. **Temps d'Entraînement**:
   - ResNet-1D Medium: ~2-3h (100 epochs, GPU)
   - ResNet-1D Large: ~5-6h (100 epochs, GPU)

3. **Early Stopping**: Le script sauvegarde le meilleur modèle automatiquement

4. **Reproductibilité**: Fixer `--seed` pour résultats reproductibles

## 🆘 Support

Pour questions/problèmes:
1. Vérifier `note 2025/TRAITEMENTS_NECESSAIRES_2025.md`
2. Vérifier `note 2025/AMELIORATIONS_PROPOSEES.md`
3. Consulter le forum PhysioNet Challenge

---

**Bon entraînement ! 🚀**
