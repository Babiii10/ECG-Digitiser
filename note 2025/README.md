# PhysioNet Challenge 2025 - Documentation Complète

**Détection de la Maladie de Chagas à partir d'ECG 12 Dérivations**

---

## 📁 Contenu de ce Répertoire

Ce répertoire contient toute la documentation nécessaire pour participer au **PhysioNet Challenge 2025** en s'appuyant sur l'infrastructure gagnante du Challenge 2024.

### 📄 Documents Disponibles

1. **[TRAITEMENTS_NECESSAIRES_2025.md](./TRAITEMENTS_NECESSAIRES_2025.md)** (Note complète principale)
   - Contexte détaillé du Challenge 2025
   - Architecture complète du pipeline à développer
   - Préparation des datasets (CODE-15%, SaMi-Trop, PTB-XL)
   - Implémentations de modèles (ResNet-1D, ViT, LSTM, Ensemble)
   - Métriques d'évaluation (TPR@5%)
   - Roadmap de développement complète
   - Checklist critique avant soumission

2. **[AMELIORATIONS_PROPOSEES.md](./AMELIORATIONS_PROPOSEES.md)** (Améliorations stratégiques)
   - 15 améliorations classées par priorité
   - Code d'implémentation pour chaque amélioration
   - Impact estimé sur TPR@5%
   - Timeline et difficulté de mise en œuvre
   - Stratégie recommandée en 4 phases

3. **[README.md](./README.md)** (Ce fichier)
   - Vue d'ensemble de la documentation
   - Quick start guide
   - Liens utiles

---

## 🎯 RÉSUMÉ EXÉCUTIF

### Challenge 2025 vs Challenge 2024

| Aspect | 2024 (Gagné ✅) | 2025 (À Faire) |
|--------|-----------------|----------------|
| **Objectif** | Numériser images ECG → signaux | Détecter maladie Chagas à partir de signaux ECG |
| **Input** | Images PNG | Signaux WFDB (12 dérivations, 400Hz) |
| **Output** | Signaux WFDB | Probabilité Chagas [0, 1] |
| **Tâche** | Segmentation + Vectorisation | Classification binaire |
| **Architecture** | nnU-Net + Hough Transform | CNN/Transformer/RNN + Ensemble |
| **Métrique** | SNR, ASCI, KS-distance | **TPR @ Top 5%** |

### 🔑 Différence Clé
**2024** : Vision par ordinateur (image → signal)
**2025** : Machine Learning sur séries temporelles (signal → label)

---

## 🚀 QUICK START

### Étape 1 : Lire la Documentation

```bash
# Commencer par le document principal
cat "note 2025/TRAITEMENTS_NECESSAIRES_2025.md"

# Puis les améliorations
cat "note 2025/AMELIORATIONS_PROPOSEES.md"
```

### Étape 2 : Télécharger les Données

```bash
# CODE-15% (>300K ECG, Brésil)
wget https://physionet.org/content/code-15/...

# SaMi-Trop (1631 ECG validés sérologiquement)
wget https://physionet.org/content/sami-trop/...

# PTB-XL (21K ECG pour pré-entraînement)
wget https://physionet.org/content/ptb-xl/1.0.3/
```

### Étape 3 : Créer l'Infrastructure

```bash
# Créer structure de dossiers
mkdir -p src_2025/{data,models,training,evaluation,utils}
mkdir -p data_2025/{CODE-15%,SaMi-Trop,PTB-XL,processed}
mkdir -p models_2025

# Copier helper_code.py (compatible WFDB)
cp src/utils/helper_code.py src_2025/utils/
```

### Étape 4 : Commencer le Développement

Suivre la **Roadmap Phase 1** (voir TRAITEMENTS_NECESSAIRES_2025.md section 9).

---

## 📊 COMPOSANTS RÉUTILISABLES DU REPO 2024

### ✅ Directement Réutilisables

| Fichier/Module | Utilité 2025 | Action |
|----------------|--------------|--------|
| `src/utils/helper_code.py` | Lecture/écriture WFDB, manipulation signaux | **Copier tel quel** |
| `config.py` | Configuration signaux (freq, units, leads) | Adapter 400Hz → 500Hz |
| `requirements.txt` (partiel) | wfdb, numpy, scipy, matplotlib | Compléter avec PyTorch |

### ❌ Non Réutilisables

- `nnUNet/` - Segmentation (pas applicable à classification)
- `src/run/digitize.py` - Hough Transform + vectorisation (pas applicable)
- `ecg-image-generator/` - Génération images synthétiques (pas nécessaire)

### 💡 Philosophie à Conserver

✅ Architecture modulaire
✅ Scripts batch automatisés
✅ Validation rigoureuse
✅ Documentation exhaustive

---

## 🎯 OBJECTIFS DE PERFORMANCE

### Benchmarks TPR@5%

- **Baseline (modèle simple)** : 0.30-0.40
- **Bon modèle (1 architecture optimisée)** : 0.50-0.60
- **Très bon modèle (ensemble)** : 0.65-0.70
- **🏆 Top 3 Challenge** : > 0.70

### Améliorations Prioritaires (Impact Majeur)

1. **Pré-entraînement PTB-XL** : +8-15% TPR@5%
2. **Architecture Multi-Échelle** : +5-10% TPR@5%
3. **Ensemble Avancé** : +5-8% TPR@5%
4. **Loss Optimisée Ranking** : +3-7% TPR@5%
5. **Data Augmentation ECG** : +4-6% TPR@5%

**Gain Total Estimé** : **+25-35% TPR@5%** (baseline → solution optimale)

---

## 📈 MÉTRIQUE PRINCIPALE : TPR @ Top 5%

### Définition

**True Positive Rate (TPR)** parmi les **top 5%** des patients classés par probabilité prédite.

### Pourquoi cette métrique ?

- Simule contrainte réelle : **capacité limitée de tests sérologiques** au Brésil (~5%)
- Question clinique : "Parmi les 5% de patients que je peux tester, combien sont vraiment positifs ?"
- Priorisation patients pour tests de confirmation

### Implémentation

```python
def compute_tpr_at_top_5_percent(y_true, y_pred_proba):
    n = len(y_true)
    n_top_5pct = max(1, int(n * 0.05))

    # Trier par probabilité décroissante
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    top_5pct_idx = sorted_idx[:n_top_5pct]

    # TPR
    n_positives_total = y_true.sum()
    n_positives_top_5pct = y_true[top_5pct_idx].sum()

    return n_positives_top_5pct / n_positives_total
```

---

## 🗓️ TIMELINE RECOMMANDÉE (10 Semaines)

### Phase 1 : Infrastructure (Semaines 1-2)
- Téléchargement données
- Prétraitement signaux
- DataLoader PyTorch
- Baseline ResNet-1D

**Livrable** : Modèle baseline fonctionnel, TPR@5% ~ 0.35

---

### Phase 2 : Performance (Semaines 3-7)
- Pré-entraînement PTB-XL
- Architecture multi-échelle
- Feature engineering clinique
- Loss optimisée + Calibration

**Livrable** : Modèle optimisé, TPR@5% ~ 0.60

---

### Phase 3 : Excellence (Semaines 8-9)
- Ensemble de modèles
- Test-time augmentation
- Hyperparameter tuning
- Post-processing avancé

**Livrable** : Solution complète, TPR@5% ~ 0.70

---

### Phase 4 : Soumission (Semaine 10)
- Validation finale
- Documentation
- Code cleanup
- Soumission officielle

**Livrable** : Soumission conforme + papier CinC 2025

---

## 📚 RESSOURCES ESSENTIELLES

### 🔗 Liens Officiels

- **Challenge 2025** : https://moody-challenge.physionet.org/2025/
- **Papier Challenge** : https://arxiv.org/abs/2510.02202
- **Code Exemple Python** : https://github.com/physionetchallenges/python-example-2025
- **Code Évaluation** : https://github.com/physionetchallenges/evaluation-2025
- **Forum** : https://groups.google.com/g/physionet-challenges

### 📊 Datasets

- **CODE-15%** : https://physionet.org/content/code-15/
- **SaMi-Trop** : https://physionet.org/content/sami-trop/
- **PTB-XL** : https://physionet.org/content/ptb-xl/1.0.3/

### 📖 Papers Importants

1. **Challenge 2024 Winner** (notre solution) : https://arxiv.org/abs/2410.14185
2. **Challenge 2025** : https://arxiv.org/abs/2510.02202
3. **Chagas Disease ECG Patterns** : Rechercher "RBBB Chagas disease"
4. **Deep Learning ECG** : "Cardiologist-level arrhythmia detection" (Rajpurkar, 2017)

---

## 🛠️ STACK TECHNOLOGIQUE RECOMMANDÉ

### Core ML
```python
torch           # PyTorch 2.0+
torchvision     # Transforms
timm            # PyTorch Image Models
transformers    # Hugging Face (ViT)
```

### Signal Processing
```python
wfdb            # WFDB format
scipy           # Filtering
neurokit2       # ECG feature extraction
```

### Evaluation
```python
scikit-learn    # Metrics
numpy
pandas
```

### MLOps
```python
wandb           # Experiment tracking
optuna          # Hyperparameter tuning
```

### Visualization
```python
matplotlib
seaborn
plotly
```

---

## ⚠️ PIÈGES À ÉVITER

### 1. Overfitting sur SaMi-Trop
- **Problème** : Seulement 1631 ECG
- **Solution** : Cross-validation 5-fold + Early stopping

### 2. Labels Bruités CODE-15%
- **Problème** : Auto-rapportés, non validés
- **Solution** : Pré-entraîner sur PTB-XL, fine-tuner sur SaMi-Trop d'abord

### 3. Optimiser Mauvaise Métrique
- **Problème** : Accuracy ≠ TPR@5%
- **Solution** : Loss customisée pour ranking

### 4. Data Leakage
- **Problème** : Fuites train/val/test
- **Solution** : Stratification stricte + validation externe

### 5. Distributional Shift
- **Problème** : Train (Brésil 2010-2016) ≠ Test (inconnu)
- **Solution** : Domain adaptation + Augmentation robuste

---

## 💡 CONSEILS PRATIQUES

### Do's ✅

1. **Commencer simple** : Baseline fonctionnel d'abord
2. **Itérer vite** : Tester idées sur subset (10% données)
3. **Valider rigoureusement** : 5-fold CV obligatoire
4. **Analyser erreurs** : Comprendre échecs du modèle
5. **Documenter tout** : Expériences, hyperparams, résultats
6. **Checkpoint régulier** : Sauvegarder modèles intermédiaires
7. **Collaborer** : Partager idées sur forum officiel

### Don'ts ❌

1. **Pas de complexité prématurée** : Simple > Complexe si même perf
2. **Pas de tout ré-entraîner** : Réutiliser checkpoints
3. **Pas négliger calibration** : Crucial pour ranking
4. **Pas ignorer features cliniques** : Domain knowledge = avantage
5. **Pas sur-optimiser validation** : Risque overfitting
6. **Pas oublier test-time augmentation** : 1-3% gain gratuit
7. **Pas procrastiner documentation** : Nécessaire pour soumission

---

## 🏆 OBJECTIF FINAL

### Vision
**Top 3 du PhysioNet Challenge 2025** avec solution robuste, reproductible et cliniquement validée.

### Critères de Succès

1. ✅ **Performance** : TPR@5% > 0.70 sur test set caché
2. ✅ **Code** : Open-source, bien documenté, reproductible
3. ✅ **Papier** : Accepté à Computing in Cardiology 2025
4. ✅ **Impact** : Solution déployable dans contexte clinique réel

---

## 📞 PROCHAINES ÉTAPES

### Immédiat (Aujourd'hui)

1. ✅ Lire [TRAITEMENTS_NECESSAIRES_2025.md](./TRAITEMENTS_NECESSAIRES_2025.md)
2. ✅ Lire [AMELIORATIONS_PROPOSEES.md](./AMELIORATIONS_PROPOSEES.md)
3. ✅ Télécharger exemple code : `git clone https://github.com/physionetchallenges/python-example-2025`

### Court Terme (Semaine 1)

1. ⬜ Télécharger datasets
2. ⬜ Créer structure de dossiers `src_2025/`
3. ⬜ Implémenter prétraitement de base
4. ⬜ Créer DataLoader PyTorch

### Moyen Terme (Semaines 2-3)

1. ⬜ Entraîner baseline ResNet-1D
2. ⬜ Implémenter métrique TPR@5%
3. ⬜ Valider sur SaMi-Trop
4. ⬜ Analyser premiers résultats

---

## 📝 NOTES

- **Deadline Phase Officielle** : Vérifier sur https://moody-challenge.physionet.org/2025/
- **Computing in Cardiology 2025** : Abstracts soumis en Juin 2025
- **Ressources GPU** : Prévoir accès GPU (cloud ou local) pour entraînement

---

## ✉️ CONTACT & SUPPORT

- **Forum Officiel** : https://groups.google.com/g/physionet-challenges
- **Issues GitHub** : Créer issue dans repo si bugs

---

**Bonne chance pour le Challenge 2025 ! 🚀**

*Remember: "In God we trust, all others must bring data." - W. Edwards Deming*

---

_Créé : Janvier 2026_
_Basé sur : Solution gagnante PhysioNet Challenge 2024_
_Objectif : Top 3 PhysioNet Challenge 2025_
