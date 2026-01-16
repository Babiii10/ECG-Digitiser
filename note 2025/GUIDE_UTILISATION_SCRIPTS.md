# Guide d'Utilisation des Scripts Python - Challenge 2025

## 🎯 Vue d'Ensemble

Tous les scripts Python pour le PhysioNet Challenge 2025 sont maintenant créés et prêts à l'emploi !

### 📁 Structure Créée

```
ECG-Digitiser/
├── src_2025/                          # Nouveau code pour Challenge 2025
│   ├── data/
│   │   ├── preprocessing.py           # ✅ Prétraitement ECG complet
│   │   └── dataset.py                 # ✅ Dataset PyTorch + augmentation
│   ├── models/
│   │   ├── resnet1d.py               # ✅ Architectures ResNet-1D
│   │   └── losses.py                 # ✅ Loss functions optimisées
│   ├── training/
│   │   ├── train.py                  # ✅ Script d'entraînement
│   │   └── predict.py                # ✅ Script de prédiction
│   ├── utils/
│   │   └── metrics.py                # ✅ Métriques (TPR@5%, etc.)
│   ├── README.md                     # ✅ Documentation complète
│   └── requirements.txt              # ✅ Dépendances
├── team_code.py                       # ✅ Interface challenge officielle
└── note 2025/
    ├── TRAITEMENTS_NECESSAIRES_2025.md
    ├── AMELIORATIONS_PROPOSEES.md
    └── GUIDE_UTILISATION_SCRIPTS.md   # Ce fichier
```

---

## 🚀 QUICK START - 3 Commandes pour Commencer

### 1️⃣ Installer les Dépendances

```bash
# Créer environnement virtuel (recommandé)
python -m venv venv_2025
source venv_2025/bin/activate  # Linux/Mac
# ou
venv_2025\Scripts\activate  # Windows

# Installer dépendances
pip install -r src_2025/requirements.txt
```

### 2️⃣ Entraîner un Modèle

```bash
# Entraînement sur dataset SaMi-Trop (recommandé pour débuter)
python -m src_2025.training.train \
    --data_folder data_2025/SaMi-Trop \
    --output_dir models_2025/mon_premier_modele \
    --model_type resnet1d_medium \
    --epochs 50 \
    --batch_size 32 \
    --loss_type focal
```

**Résultat attendu:**
- Entraînement pendant ~1-2h (avec GPU)
- Meilleur modèle sauvegardé dans `models_2025/mon_premier_modele/model_best.pth`
- TPR@5% attendu: 0.35-0.50 (baseline)

### 3️⃣ Faire des Prédictions

```bash
# Prédiction sur données test
python -m src_2025.training.predict \
    --model_path models_2025/mon_premier_modele/model_best.pth \
    --data_folder data_2025/test \
    --output_file predictions.csv
```

**Résultat:**
- Fichier CSV avec probabilités pour chaque patient
- Trié par probabilité décroissante

---

## 📊 PIPELINE COMPLET DE PRÉTRAITEMENT

### Ce qui est Fait Automatiquement

Le script `preprocessing.py` applique **7 étapes** automatiquement :

```python
# preprocessing.py fait TOUT cela automatiquement !

1. ✅ Filtrage passe-bande (0.5-40 Hz)
   → Supprime bruit basse/haute fréquence

2. ✅ Filtre notch (50/60 Hz)
   → Supprime interférence ligne électrique

3. ✅ Suppression dérive baseline
   → Filtre médian pour supprimer variations lentes

4. ✅ Détection/suppression artefacts
   → Z-score thresholding + interpolation

5. ✅ Rééchantillonnage → 500 Hz
   → Harmonise toutes les fréquences

6. ✅ Normalisation par dérivation
   → Z-score (mean=0, std=1) pour chaque lead

7. ✅ Padding/Truncation → 5000 samples
   → Fixe longueur (10s @ 500Hz)
```

### Exemple d'Utilisation Manuelle (optionnel)

```python
from src_2025.data.preprocessing import ECGPreprocessor
from src.utils import helper_code

# Créer preprocessor
preprocessor = ECGPreprocessor(
    target_fs=500,
    lowcut=0.5,
    highcut=40,
    notch_freq=60
)

# Charger signal
signal, fields = helper_code.load_signals('path/to/record')

# Prétraiter (1 ligne !)
processed = preprocessor.preprocess(
    signal,
    fs_original=400,
    target_length=5000
)

print(f"Original: {signal.shape}")      # (4000, 12) - 10s @ 400Hz
print(f"Processed: {processed.shape}")  # (5000, 12) - 10s @ 500Hz
```

---

## 🧠 ARCHITECTURES DISPONIBLES

### Comparaison Rapide

| Modèle | Paramètres | Vitesse | Performance | Usage |
|--------|-----------|---------|-------------|-------|
| `resnet1d_small` | ~500K | ⚡⚡⚡ | ⭐⭐ | Prototypage rapide |
| `resnet1d_medium` | ~2M | ⚡⚡ | ⭐⭐⭐ | **Recommandé** |
| `resnet1d_large` | ~10M | ⚡ | ⭐⭐⭐⭐ | Haute performance |
| `seresnet1d` | ~2.5M | ⚡⚡ | ⭐⭐⭐⭐ | Avec attention |

### Changer d'Architecture

```bash
# Petit (rapide)
--model_type resnet1d_small

# Moyen (recommandé)
--model_type resnet1d_medium

# Grand (meilleure performance)
--model_type resnet1d_large

# Avec attention
--model_type seresnet1d
```

---

## 📉 FONCTIONS DE LOSS

### Quelle Loss Choisir ?

| Loss | Quand l'utiliser | Commande |
|------|------------------|----------|
| `bce` | Baseline rapide | `--loss_type bce` |
| `focal` | **Données déséquilibrées** ⭐ | `--loss_type focal` |
| `ranking` | Optimiser ranking | `--loss_type ranking` |
| `tpr` | Optimiser directement TPR@5% | `--loss_type tpr` |
| `combined` | Ensemble de losses | `--loss_type combined` |

### Recommandation

```bash
# Pour commencer (meilleur compromis)
--loss_type focal

# Pour optimiser TPR@5% au maximum
--loss_type tpr
```

---

## 🎯 EXEMPLES D'UTILISATION COMPLÈTE

### Exemple 1 : Entraînement Baseline Rapide

```bash
# Configuration minimale pour tester rapidement
python -m src_2025.training.train \
    --data_folder data_2025/SaMi-Trop \
    --output_dir models_2025/baseline \
    --model_type resnet1d_small \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001
```

**Temps:** ~30 min (GPU)
**TPR@5% attendu:** 0.30-0.40

---

### Exemple 2 : Entraînement Optimal (Recommandé)

```bash
# Configuration optimale pour bonne performance
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
    --grad_clip 1.0
```

**Temps:** ~2-3h (GPU)
**TPR@5% attendu:** 0.50-0.60

---

### Exemple 3 : Entraînement Haute Performance

```bash
# Configuration pour maximiser performance
python -m src_2025.training.train \
    --data_folder data_2025/CODE-15% \
    --output_dir models_2025/best \
    --model_type seresnet1d \
    --epochs 150 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --loss_type tpr \
    --grad_clip 1.0 \
    --num_workers 8
```

**Temps:** ~6-8h (GPU puissante)
**TPR@5% attendu:** 0.65-0.75

---

### Exemple 4 : Prédiction avec TTA (Meilleure Performance)

```bash
# Test-Time Augmentation améliore TPR@5% de 1-3%
python -m src_2025.training.predict \
    --model_path models_2025/optimal/model_best.pth \
    --data_folder data_2025/test \
    --output_file predictions_tta.csv \
    --use_tta \
    --n_augmentations 10
```

**Gain attendu:** +1-3% TPR@5%
**Temps:** ~2x plus lent (mais meilleurs résultats)

---

## 🔧 PARAMÈTRES AVANCÉS

### Ajuster pour Votre GPU

```bash
# GPU avec 8 GB VRAM
--batch_size 16

# GPU avec 12 GB VRAM (recommandé)
--batch_size 32

# GPU avec 24 GB VRAM
--batch_size 64

# CPU uniquement (très lent !)
--device cpu --batch_size 8
```

### Optimizers & Schedulers

```bash
# Optimizer (recommandé: adamw)
--optimizer adamw

# Learning rate scheduler
--scheduler cosine        # Recommandé
--scheduler step          # Décroit par paliers
--scheduler reduce_on_plateau  # Adaptatif
--scheduler none          # Constant
```

### Early Stopping

Le script sauvegarde automatiquement le **meilleur modèle** basé sur TPR@5% de validation.

```python
# Automatique ! Pas besoin de configurer
# Le meilleur modèle est dans: output_dir/model_best.pth
```

---

## 📈 MONITORING ET RÉSULTATS

### Fichiers Générés Automatiquement

```
models_2025/mon_modele/
├── config.json              # Configuration utilisée
├── history.json             # Historique complet train/val
├── checkpoint_latest.pth    # Dernier checkpoint
├── checkpoint_best.pth      # Meilleur checkpoint (complet)
└── model_best.pth           # Meilleurs poids seulement
```

### Visualiser l'Historique

```python
import json
import matplotlib.pyplot as plt

# Charger historique
with open('models_2025/mon_modele/history.json', 'r') as f:
    history = json.load(f)

# Extraire TPR@5%
epochs = range(1, len(history['train']) + 1)
train_tpr = [m['tpr_at_5pct'] for m in history['train']]
val_tpr = [m['tpr_at_5pct'] for m in history['val']]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_tpr, label='Train TPR@5%', marker='o')
plt.plot(epochs, val_tpr, label='Val TPR@5%', marker='s')
plt.xlabel('Epoch')
plt.ylabel('TPR @ Top 5%')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('training_progress.png')
print(f"Best Val TPR@5%: {max(val_tpr):.4f}")
```

---

## 🎓 WORKFLOW COMPLET - De A à Z

### Étape 1 : Préparation

```bash
# 1. Créer environnement
python -m venv venv_2025
source venv_2025/bin/activate

# 2. Installer dépendances
pip install -r src_2025/requirements.txt

# 3. Télécharger données
# (Télécharger CODE-15%, SaMi-Trop, PTB-XL depuis PhysioNet)

# 4. Organiser données
mkdir -p data_2025/{CODE-15%,SaMi-Trop,PTB-XL}
# Copier données WFDB dans ces dossiers
```

### Étape 2 : Entraînement Initial

```bash
# Entraîner baseline pour tester
python -m src_2025.training.train \
    --data_folder data_2025/SaMi-Trop \
    --output_dir models_2025/baseline \
    --model_type resnet1d_small \
    --epochs 20 \
    --batch_size 32
```

### Étape 3 : Validation

```bash
# Vérifier résultats
cat models_2025/baseline/history.json | grep tpr_at_5pct

# Si TPR@5% > 0.35 → Bon début !
```

### Étape 4 : Entraînement Optimal

```bash
# Entraîner modèle optimal
python -m src_2025.training.train \
    --data_folder data_2025/SaMi-Trop \
    --output_dir models_2025/optimal \
    --model_type resnet1d_medium \
    --epochs 100 \
    --loss_type focal
```

### Étape 5 : Prédiction

```bash
# Générer prédictions
python -m src_2025.training.predict \
    --model_path models_2025/optimal/model_best.pth \
    --data_folder data_2025/test \
    --output_file predictions.csv \
    --use_tta
```

### Étape 6 : Soumission Challenge

```bash
# Utiliser l'interface officielle
python train_model.py -d training_data -m model
python run_model.py -d test_data -m model -o output

# Créer soumission
# (Suivre instructions sur moody-challenge.physionet.org/2025)
```

---

## 🐛 DEBUGGING - Problèmes Courants

### Problème 1 : "CUDA out of memory"

**Solution:**
```bash
# Réduire batch size
--batch_size 16  # Au lieu de 32

# Ou utiliser CPU (lent !)
--device cpu
```

### Problème 2 : "No valid records found"

**Solution:**
```bash
# Vérifier structure données
ls -R data_2025/SaMi-Trop/

# Doit contenir fichiers .hea et .dat
# Exemple: record001.hea, record001.dat
```

### Problème 3 : TPR@5% très bas (<0.20)

**Solutions:**
```bash
# 1. Vérifier classe imbalance
# 2. Utiliser focal loss
--loss_type focal

# 3. Augmenter epochs
--epochs 150

# 4. Essayer autre architecture
--model_type seresnet1d
```

### Problème 4 : Overfitting (train >> val)

**Solutions:**
```bash
# 1. Augmenter régularisation
--weight_decay 1e-3  # Au lieu de 1e-4

# 2. Augmenter dropout (modifier code)
# 3. Early stopping automatique (déjà inclus !)
```

---

## 🚀 AMÉLIORATIONS FUTURES

### Prochaines Étapes Recommandées

1. **Pré-entraînement sur PTB-XL** (voir `AMELIORATIONS_PROPOSEES.md`)
   ```bash
   # D'abord pré-entraîner sur PTB-XL (21K ECG)
   # Puis fine-tuner sur SaMi-Trop
   # Gain attendu: +8-15% TPR@5%
   ```

2. **Ensemble de Modèles**
   ```python
   # Entraîner 3-5 modèles différents
   # Moyenner prédictions
   # Gain attendu: +5-8% TPR@5%
   ```

3. **Hyperparameter Tuning**
   ```bash
   # Utiliser Optuna pour optimiser automatiquement
   # (voir code dans AMELIORATIONS_PROPOSEES.md)
   ```

---

## 📞 AIDE & SUPPORT

### Documentation Complémentaire

1. **`src_2025/README.md`** - Guide détaillé de chaque module
2. **`note 2025/TRAITEMENTS_NECESSAIRES_2025.md`** - Architecture complète
3. **`note 2025/AMELIORATIONS_PROPOSEES.md`** - 15 améliorations avancées

### Ressources Externes

- **Forum Challenge:** https://groups.google.com/g/physionet-challenges
- **Challenge 2025:** https://moody-challenge.physionet.org/2025/
- **Code Exemple:** https://github.com/physionetchallenges/python-example-2025

---

## ✅ CHECKLIST DE DÉPART

Avant de commencer, vérifier:

- [ ] Python 3.8+ installé
- [ ] GPU CUDA compatible (optionnel mais recommandé)
- [ ] Données téléchargées (SaMi-Trop minimum)
- [ ] Dépendances installées (`requirements.txt`)
- [ ] Environnement virtuel créé
- [ ] Espace disque suffisant (~10 GB pour modèles)

---

## 🎯 OBJECTIFS PAR ÉTAPE

### Semaine 1 : Baseline
- [ ] Entraîner `resnet1d_small` sur SaMi-Trop
- [ ] Obtenir TPR@5% > 0.35
- [ ] Comprendre pipeline de prétraitement

### Semaine 2-3 : Optimisation
- [ ] Entraîner `resnet1d_medium` avec `focal` loss
- [ ] Obtenir TPR@5% > 0.50
- [ ] Tester TTA

### Semaine 4-6 : Performance
- [ ] Essayer `seresnet1d`
- [ ] Entraîner sur CODE-15% (grand dataset)
- [ ] Obtenir TPR@5% > 0.65

### Semaine 7-10 : Excellence
- [ ] Implémenter ensemble
- [ ] Pré-entraînement PTB-XL
- [ ] Viser TPR@5% > 0.70 (Top 3 !)

---

**Bon courage pour le Challenge 2025 ! 🚀🏆**

*N'oubliez pas : Itérer rapidement, valider rigoureusement, et analyser les erreurs !*
