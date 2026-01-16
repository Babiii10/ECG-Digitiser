# Notebooks ECG Digitiser

Ce dossier contient les notebooks Jupyter pour le projet ECG Digitiser.

## Notebooks Disponibles

### 01_preprocessing_setup.ipynb
**Preprocessing & Configuration des Données**

Ce notebook contient tous les prérequis pour le traitement des données ECG du challenge PhysioNet.

#### Fonctionnalités:

1. **Configuration de l'Environnement**
   - Import de toutes les bibliothèques nécessaires
   - Configuration des chemins de données (Kaggle et local)
   - Détection automatique de l'environnement
   - Configuration du device (CPU/GPU)

2. **Exploration des Données**
   - Chargement des métadonnées (train.csv, test.csv)
   - Analyse de la structure des dossiers
   - Statistiques sur les données

3. **Gestion Train/Validation**
   - Split train/validation avec stratification par patient
   - Validation croisée K-Fold
   - Prévention de la fuite de données (data leakage)

4. **Classes Dataset PyTorch**
   - `ECGImageDataset`: Dataset pour les images ECG
   - `ECGSegmentationDataset`: Dataset pour la segmentation avec masks
   - Support des transformations personnalisées

5. **Augmentation de Données**
   - Transformations d'entraînement (rotation, translation, couleur)
   - Transformations de validation/test
   - Normalisation ImageNet

6. **DataLoaders**
   - Configuration optimisée pour l'entraînement
   - Support multi-workers
   - Pin memory pour GPU

7. **Visualisation**
   - Affichage des images ECG
   - Visualisation des batches
   - Distribution des données
   - Historique d'entraînement

8. **Preprocessing Avancé**
   - Amélioration de contraste (CLAHE)
   - Débruitage
   - Extraction des lignes de grille
   - Suppression de la grille
   - Normalisation des signaux

#### Utilisation:

```python
# Le notebook crée automatiquement les variables suivantes:

# Splits de données
train_samples  # Liste d'échantillons d'entraînement
val_samples    # Liste d'échantillons de validation
all_test_samples  # Liste d'échantillons de test

# DataLoaders prêts à l'emploi
train_loader   # DataLoader d'entraînement
val_loader     # DataLoader de validation
test_loader    # DataLoader de test

# K-Fold pour validation croisée
kfold_splits   # Liste de tuples (train, val) pour chaque fold

# Configuration
device         # torch.device (cuda ou cpu)
BATCH_SIZE     # Taille des batchs
IMG_SIZE       # Taille des images
```

#### Configuration:

Les paramètres principaux sont:
- `BATCH_SIZE = 8`: Taille des batches
- `IMG_SIZE = (256, 256)`: Taille des images après redimensionnement
- `VALIDATION_SPLIT = 0.2`: Proportion de validation (20%)
- `N_FOLDS = 5`: Nombre de folds pour la validation croisée
- `SEED = 42`: Seed pour la reproductibilité

#### Sorties:

Le notebook génère:
- `outputs/preprocessing_config.json`: Configuration sauvegardée
- Variables Python pour utilisation immédiate dans les notebooks suivants

## Structure des Données Attendue

```
data/physionet-ecg-image-digitization/
├── train/
│   ├── patient_id_1/
│   │   ├── patient_id_1.csv
│   │   ├── patient_id_1-0001.png
│   │   ├── patient_id_1-0002.png
│   │   └── ...
│   ├── patient_id_2/
│   └── ...
├── test/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
├── train.csv
├── test.csv
└── sample_submission.parquet
```

## Prochains Notebooks

Les notebooks suivants seront créés pour:
- **02_model_training.ipynb**: Entraînement du modèle de segmentation
- **03_inference.ipynb**: Inférence et génération de prédictions
- **04_evaluation.ipynb**: Évaluation et analyse des résultats

## Environnement

### Prérequis

Les packages nécessaires sont listés dans `requirements.txt`:
- PyTorch
- torchvision
- OpenCV
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm

### Installation

```bash
pip install -r requirements.txt
```

### Pour Kaggle

Le notebook détecte automatiquement l'environnement Kaggle et ajuste les chemins en conséquence.

## Notes

- Le notebook est conçu pour fonctionner aussi bien en local que sur Kaggle
- Toutes les fonctions sont documentées avec des docstrings
- Les visualisations sont optimisées pour l'analyse exploratoire
- Le split train/val garantit qu'aucun patient n'apparaît dans les deux ensembles
