# Projet de Détection de Pneumonie

Projet réalisé dans le cadre de mes études à Epitech.
Ce projet vise à développer un système de détection automatique de pneumonie à partir de radiographies thoraciques en utilisant des techniques d'apprentissage automatique.

## Structure du Projet

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
│   ├── models/                # Modèles d'apprentissage automatique
│   │   └── predict.py         # Script pour charger le modèle et faire des prédictions
│   │
│   └── utils/                 # Utilitaires divers
│
├── notebooks/                 # Notebooks Jupyter
│   ├── 01_exploration.ipynb   # Exploration des données
│   ├── 02_preprocessing.ipynb # Test du prétraitement
│   ├── 03_model_training_sklearn_with_pca.ipynb # Entraînement des modèles avec PCA
│   └── 04_model_training_sklearn.ipynb # Entraînement des modèles avec et sans PCA
│
├── models/                    # Modèles entraînés sauvegardés (exclus de Git en raison de leur taille)
│
├── tests/                     # Tests unitaires
│   ├── test_preprocessing.py  # Tests des fonctions de prétraitement
│   ├── test_predict.py        # Tests des fonctions de prédiction
│   └── test_visualizer.py     # Tests des fonctions de visualisation
│
├── demo.py                    # Script de démonstration
├── index.html                 # Page de présentation du projet
├── rapport_pneumonie.pdf      # Rapport des résultats du modèle
├── T-DEV-810_project.pdf      # Document du projet
└── requirements.txt           # Dépendances du projet
```

## Installation

Pour installer les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Exploration et prétraitement des données

- **Exploration des données** : Consultez le notebook `notebooks/01_exploration.ipynb`
- **Prétraitement des images** : Consultez le notebook `notebooks/02_preprocessing.ipynb`

### 2. Entraînement des modèles

#### Notebook 03 : Entraînement avec PCA

Le notebook `notebooks/03_model_training_sklearn_with_pca.ipynb` contient l'entraînement de plusieurs modèles d'apprentissage automatique avec réduction de dimension par PCA. Les modèles testés incluent :
- Régression Logistique
- Arbre de Décision
- Random Forest
- SVM

#### Notebook 04 : Entraînement avec et sans PCA

Le notebook `notebooks/04_model_training_sklearn.ipynb` étend l'analyse en :
- Entraînant chaque modèle avec et sans PCA (possibilité d'activer/desactiver le PCA)
- Testant différentes configurations de PCA (10, 20, 50, 100, 1000 composantes)
- Ajoutant un tableau comparatif des scores de tous les modèles à la fin

## Résultats

Les résultats des modèles sont maintenant présentés avec un formatage amélioré :
- Les métriques (accuracy, precision, recall, F1, ROC AUC) sont affichées en pourcentage avec 2 décimales
- Les temps d'exécution sont arrondis à 3 décimales

Le modèle de **SVM** avec la configuration PCA-1000 a obtenu les meilleures performances avec :
- **Exactitude (Accuracy)** : 97,39%
- **Précision (Precision)** : 98,35%
- **Rappel (Recall)** : 98,14%
- **Score F1** : 98,24%
- **ROC AUC** : 99,54%

## Tests unitaires

Pour exécuter les tests unitaires :

```bash
# Exécuter tous les tests
python -m unittest discover tests

# Exécuter un test spécifique
python -m unittest tests/test_predict.py
python -m unittest tests/test_preprocessing.py
python -m unittest tests/test_visualizer.py
```

## Limitations et perspectives

- Le modèle actuel ne peut pas distinguer entre différents types de pneumonie (virale, bactérienne)
- La qualité des radiographies peut affecter les performances du modèle
- Les travaux futurs pourraient inclure l'utilisation de réseaux de neurones plus avancés
