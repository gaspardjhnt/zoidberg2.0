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
│   ├── 04_model_training_sklearn.ipynb # Entraînement des modèles avec et sans PCA
│   └── 05_model_training_sklearn_different_scoring.ipynb # Test de différentes métriques de scoring
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

#### Notebook 05 : Test de différentes métriques de scoring

Le notebook `notebooks/05_model_training_sklearn_different_scoring.ipynb` pousse l'analyse plus loin en :
- Testant différentes métriques de scoring pour l'hyperparamétrage des modèles (accuracy, precision, recall, f1, roc_auc)
- Comparant l'impact du choix de la métrique de scoring sur les performances finales des modèles
- Visualisant les résultats par métrique de scoring et par modèle
- Identifiant la meilleure combinaison modèle/configuration/métrique de scoring

## Résultats

### Formatage des résultats

Les résultats des modèles sont présentés avec un formatage amélioré :
- Les métriques (accuracy, precision, recall, F1, ROC AUC) sont affichées en pourcentage avec 2 décimales
- Les temps d'exécution sont arrondis à 3 décimales

### Meilleur modèle (Notebook 04)

Le modèle de **SVM** avec la configuration PCA-1000 a obtenu les meilleures performances avec :
- **Exactitude (Accuracy)** : 97,39%
- **Précision (Precision)** : 98,35%
- **Rappel (Recall)** : 98,14%
- **Score F1** : 98,24%
- **ROC AUC** : 99,54%

### Impact des métriques de scoring (Notebook 05)

L'analyse des différentes métriques de scoring a révélé que :

- Le choix de la métrique de scoring influence significativement la sélection des hyperparamètres optimaux
- Les modèles optimisés pour le rappel (recall) tendent à avoir une meilleure sensibilité mais parfois au détriment de la précision
- Les modèles optimisés pour la précision (precision) sont plus conservateurs dans leurs prédictions
- L'optimisation sur le score F1 offre généralement le meilleur équilibre entre précision et rappel
- L'optimisation sur ROC AUC produit des modèles avec une bonne capacité de discrimination globale

Le meilleur compromis a été obtenu en optimisant sur le score F1, qui a permis d'identifier le SVM avec PCA-1000 comme configuration optimale.

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
