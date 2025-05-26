# Projet de Détection de Pneumonie

Projet réalisé dans le cadre de mes études à Epitech.
Ce projet vise à développer un système de détection automatique de pneumonie à partir de radiographies thoraciques en utilisant des techniques d'apprentissage automatique.

## Structure du Projet

```
zoidberg2.0/
│
├── data/                      # Dossier de données (radiographies thoraciques)
│
├── src/                       # Code source principal
│   ├── preprocessing/         # Traitement des images
│   │   └── preprocess_images.py # Fonctions de prétraitement d'images
│   │
│   ├── visualization/         # Visualisation des données
│   │   └── visualizer.py      # Fonctions de visualisation
│   │
│   ├── models/                # Modèles d'apprentissage automatique
│   │   └── predict.py         # Script pour charger le modèle et faire des prédictions
│   │
│   └── utils/                 # Utilitaires divers
│
├── notebooks/                 # Notebooks Jupyter
│   ├── 01_exploration.ipynb   # Exploration des données
│   ├── 02_preprocessing.ipynb # Test du prétraitement
│   └── 03_model_training_sklearn_with_pca.ipynb # Entraînement des modèles avec PCA
│
├── models/                    # Modèles entraînés sauvegardés
│   └── régression_logistique_model.pkl # Modèle de régression logistique entraîné
│
├── tests/                     # Tests unitaires
│   ├── test_preprocessing.py  # Tests des fonctions de prétraitement
│   └── test_predict.py        # Tests des fonctions de prédiction
│
├── demo.py                    # Script de démonstration
├── index.html                 # Page de présentation du projet
├── rapport_pneumonie.pdf      # Rapport des résultats du modèle
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

Le notebook `notebooks/03_model_training_sklearn_with_pca.ipynb` contient l'entraînement de plusieurs modèles d'apprentissage automatique avec réduction de dimension par PCA. Les modèles testés incluent :
- Régression Logistique (meilleure performance)
- Arbre de Décision
- Random Forest
- SVM

### 3. Utilisation du modèle entraîné

Vous pouvez utiliser le script de démonstration pour faire des prédictions sur de nouvelles images :

```bash
# Pour une seule image
python demo.py chemin/vers/image.jpg

# Pour un dossier d'images
python demo.py --batch chemin/vers/dossier
```

Ou utiliser le modèle directement dans votre code :

```python
from src.models.predict import predict_pneumonia

# Prédiction sur une image
result, probability = predict_pneumonia("chemin/vers/image.jpg")
print(f"Résultat: {result}, Probabilité: {probability:.4f}")
```

## Résultats

Le modèle de **Régression Logistique** a obtenu les meilleures performances avec :
- **Exactitude (Accuracy)** : 95.94%
- **Précision (Precision)** : 97.81%
- **Rappel (Recall)** : 96.70%
- **Score F1** : 97.25%

Ces résultats montrent que le modèle est particulièrement efficace pour détecter les cas de pneumonie tout en maintenant une bonne capacité à identifier les cas normaux.

## Tests unitaires

Pour exécuter les tests unitaires :

```bash
python -m unittest tests/test_predict.py
python -m unittest tests/test_preprocessing.py
```

## Limitations et perspectives

- Le modèle actuel ne peut pas distinguer entre différents types de pneumonie (virale, bactérienne)
- La qualité des radiographies peut affecter les performances du modèle
- Les travaux futurs pourraient inclure l'utilisation de réseaux de neurones plus avancés
