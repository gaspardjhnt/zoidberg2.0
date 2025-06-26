# Détection de Pneumonie par Apprentissage Automatique

**Projet T-DEV-810 - Juin 2025**

Ce projet vise à développer un système de détection automatique de pneumonie à partir de radiographies thoraciques en comparant systématiquement les approches classiques d'apprentissage automatique et les modèles de deep learning.

## 🎯 Objectifs du Projet

- Comparer les performances des modèles classiques (sklearn) et des modèles de deep learning
- Diagnostiquer les problèmes d'overfitting des modèles classiques
- Développer un système robuste de détection de pneumonie
- Analyser l'impact de différentes techniques (PCA, augmentation de données, transfer learning)

## 📊 Résultats Principaux

### ❌ Échec des Modèles Classiques : Overfitting Massif

Les modèles classiques (régression logistique, SVM, random forest) ont révélé un problème majeur :
- **Performances sur validation** : 96-97% d'accuracy, 97-98% de F1-score
- **Performances sur test** : 74-77% d'accuracy, 81-84% de F1-score
- **Écart moyen** : 20% de chute de performance

### ✅ Succès des Modèles Deep Learning : Robustesse et Généralisation

Les modèles de deep learning ont démontré une robustesse exceptionnelle :
- **Performances stables** : 93-95% d'accuracy sur le test
- **Écart minimal** : Seulement 1% entre validation et test
- **Meilleur modèle** : ResNet50 avec 95,19% d'accuracy et 96,12% de F1-score

## 🏗️ Structure du Projet

```
zoidberg2.0/
│
├── data/                      # Dossier de données (radiographies thoraciques)
│   └── chest_Xray/            # Images de radiographies thoraciques
│
├── src/                       # Code source principal
│   ├── preprocessing/         # Traitement des images
│   │   └── preprocess_images.py # Fonctions de prétraitement d'images
│   │
│   ├── visualization/         # Visualisation des données
│   │   └── visualizer.py      # Fonctions de visualisation pour les tableaux de métriques
│   │
│   └── models/                # Modèles d'apprentissage automatique
│       └── predict.py         # Script pour charger le modèle et faire des prédictions
│
├── notebooks/                 # Notebooks Jupyter
│   ├── 01_exploration.ipynb   # Exploration des données
│   ├── 02_preprocessing.ipynb # Test du prétraitement
│   ├── 03_model_training_sklearn_with_pca.ipynb # Entraînement des modèles avec PCA
│   ├── 04_model_training_sklearn.ipynb # Entraînement des modèles avec et sans PCA
│   └── 05_model_training_sklearn_different_scoring.ipynb # Test de différentes métriques de scoring
│
├── analyze/                   # Scripts d'analyse complémentaire
│   ├── analyze_data_distribution.py
│   ├── analyze_split_clustering.py
│   ├── analyze_test_errors.py
│   ├── analyze_overfitting_report.py
│   └── img/                   # Figures générées par les analyses
│
├── models/                    # Modèles entraînés sauvegardés
├── tests/                     # Tests unitaires
├── repport.html              # Rapport complet du projet
└── requirements.txt           # Dépendances du projet
```

## 🔬 Méthodologie

### Modèles Classiques Testés
- **Régression Logistique** : Modèle linéaire simple
- **Arbre de Décision** : Modèle non-linéaire
- **Random Forest** : Ensemble d'arbres de décision
- **SVM** : Support Vector Machine

### Modèles Deep Learning Testés
- **CNN personnalisé** : Architecture convolutive spécifique
- **VGG16** : Transfer learning depuis ImageNet
- **ResNet50** : Architecture résiduelle avec transfer learning

### Techniques Utilisées
- **PCA** : Réduction de dimensionnalité (10, 20, 50, 100, 1000 composantes)
- **Augmentation de données** : Rotations, translations, zoom, flip
- **Transfer learning** : Utilisation de modèles pré-entraînés
- **Grad-CAM** : Interprétabilité visuelle des modèles deep learning

## 📈 Résultats Détaillés

### Comparaison Modèles Classiques vs Deep Learning

| Famille de Modèles | Accuracy (Val) | F1-score (Val) | Accuracy (Test) | F1-score (Test) | Écart Val-Test |
|-------------------|----------------|----------------|-----------------|-----------------|----------------|
| **Modèles Classiques** | 96-97% | 97-98% | 74-77% | 81-84% | 19-22% |
| **Deep Learning** | 94-96% | 94-96% | 93-95% | 94-96% | 1% |

### Meilleurs Modèles par Famille

#### Modèles Classiques
- **SVM avec PCA-1000** : 77,83% accuracy test, 84,67% F1-score
- **Random Forest avec PCA-100** : 76,52% accuracy test, 82,18% F1-score

#### Modèles Deep Learning
- **ResNet50** : 95,19% accuracy test, 96,12% F1-score
- **VGG16** : 94,71% accuracy test, 95,70% F1-score
- **CNN personnalisé** : 93,27% accuracy test, 94,44% F1-score

## 🔍 Diagnostic du Problème

L'analyse approfondie a révélé que l'échec des modèles classiques n'était pas lié à :
- ❌ Un bug de code ou d'implémentation
- ❌ Un shift de distribution entre les splits (confirmé par t-SNE)
- ❌ Une mauvaise répartition des données

Mais plutôt à la **complexité intrinsèque des images médicales** qui dépasse la capacité des modèles classiques, même avec PCA et tuning d'hyperparamètres.


## 💻 Configuration Machine

- **Carte graphique** : NVIDIA GeForce RTX 4060 Ti (8 GB)
- **Mémoire RAM** : 16 Go (3200 MHz)
- **Processeur** : AMD Ryzen 7 5700X 8-Core Processor (3.40 GHz)

## 📋 Limitations

- Classification binaire uniquement (normal vs pneumonie)
- Pas de distinction entre types de pneumonie (virale, bactérienne)
- Dépendance à la qualité des radiographies
- Généralisation limitée à d'autres populations

## 🚀 Recommandations pour la Production

**Pour un déploiement en production, nous recommandons fortement l'utilisation de modèles de deep learning**, en particulier ResNet50 ou VGG16 avec transfer learning, pour les raisons suivantes :

- ✅ **Fiabilité** : Performances stables et reproductibles
- ✅ **Robustesse** : Capacité de généralisation sur des données variées
- ✅ **Interprétabilité** : Visualisations Grad-CAM pour gagner la confiance des cliniciens
- ✅ **Évolutivité** : Possibilité d'amélioration continue avec plus de données

## 📚 Documentation

- **Rapport complet** : `repport.html` - Analyse détaillée des résultats et méthodologie
- **Présentation** : `index.html` - Page de présentation du projet
- **Document du projet** : `T-DEV-810_project.pdf` - Spécifications initiales

## 🤝 Contribution

Ce projet a été réalisé dans le cadre du cours T-DEV-810 à Epitech. Les contributions sont les bienvenues pour améliorer les performances ou ajouter de nouvelles fonctionnalités.

## 📄 Licence

Ce projet est destiné à des fins éducatives et de recherche.

---

**Note importante** : Ce système est conçu comme un outil d'aide au diagnostic et non comme un remplacement du jugement clinique. Les décisions médicales finales devraient toujours être prises par des professionnels de santé qualifiés.
 