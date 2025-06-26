"""
Module de prétraitement d'images pour la détection de pneumonie à partir de radiographies thoraciques.

Ce module contient des fonctions pour préparer les images pour l'entraînement d'un modèle
de deep learning, notamment le redimensionnement, la normalisation et l'augmentation des données.
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

def redimensionner_image(image, taille_cible=(150, 150)):
    """
    Redimensionne une image à la taille cible.
    
    Args:
        image (numpy.ndarray): Image à redimensionner.
        taille_cible (tuple): Dimensions cibles (hauteur, largeur).
        
    Returns:
        numpy.ndarray: Image redimensionnée.
    """
    return cv2.resize(image, taille_cible, interpolation=cv2.INTER_AREA)

def normaliser_image(image):
    """
    Normalise les valeurs de pixels d'une image entre 0 et 1.
    
    Args:
        image (numpy.ndarray): Image à normaliser.
        
    Returns:
        numpy.ndarray: Image normalisée.
    """
    return image / 255.0

def charger_et_pretraiter_image(chemin_image, taille_cible=(150, 150), normaliser=True, canal_gris=True):
    """
    Charge une image, la convertit en niveaux de gris si nécessaire,
    la redimensionne et la normalise.
    
    Args:
        chemin_image (str): Chemin vers l'image à charger.
        taille_cible (tuple): Dimensions cibles (hauteur, largeur).
        normaliser (bool): Si True, normalise les valeurs de pixels entre 0 et 1.
        canal_gris (bool): Si True, convertit l'image en niveaux de gris.
        
    Returns:
        numpy.ndarray: Image prétraitée.
    """
    # Charger l'image
    image = cv2.imread(chemin_image)
    
    # Convertir en niveaux de gris si nécessaire
    if canal_gris and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Redimensionner l'image
    image = redimensionner_image(image, taille_cible)
    
    # Normaliser si nécessaire
    if normaliser:
        image = normaliser_image(image)
    
    # Ajouter une dimension pour le canal si l'image est en niveaux de gris
    if canal_gris:
        image = np.expand_dims(image, axis=-1)
    
    return image

def pretraiter_lot_images(chemin_dossier, taille_cible=(150, 150), normaliser=True, canal_gris=True, limite=None):
    """
    Prétraite un lot d'images dans un dossier.
    
    Args:
        chemin_dossier (str): Chemin vers le dossier contenant les images.
        taille_cible (tuple): Dimensions cibles (hauteur, largeur).
        normaliser (bool): Si True, normalise les valeurs de pixels entre 0 et 1.
        canal_gris (bool): Si True, convertit les images en niveaux de gris.
        limite (int): Nombre maximal d'images à prétraiter (None pour toutes).
        
    Returns:
        numpy.ndarray: Tableau d'images prétraitées.
    """
    # Lister les fichiers d'images
    fichiers_images = [f for f in os.listdir(chemin_dossier) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limiter le nombre d'images si nécessaire
    if limite is not None:
        fichiers_images = fichiers_images[:limite]
    
    # Prétraiter chaque image
    images = []
    for fichier in fichiers_images:
        chemin_image = os.path.join(chemin_dossier, fichier)
        image = charger_et_pretraiter_image(chemin_image, taille_cible, normaliser, canal_gris)
        images.append(image)
    
    return np.array(images)

def augmenter_image(image, rotation_max=20, translation_max=0.1, zoom_range=0.1, flip_horizontal=True):
    """
    Applique des transformations aléatoires à une image pour l'augmentation de données.
    
    Args:
        image (numpy.ndarray): Image à augmenter.
        rotation_max (int): Angle maximal de rotation en degrés.
        translation_max (float): Translation maximale en proportion de la taille de l'image.
        zoom_range (float): Plage de zoom (1-zoom_range, 1+zoom_range).
        flip_horizontal (bool): Si True, applique un retournement horizontal aléatoire.
        
    Returns:
        numpy.ndarray: Image augmentée.
    """
    # Obtenir les dimensions de l'image
    height, width = image.shape[:2]
    
    # Supprimer la dimension du canal si présente
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze()
    
    # Rotation aléatoire
    if rotation_max > 0:
        angle = random.uniform(-rotation_max, rotation_max)
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
    
    # Translation aléatoire
    if translation_max > 0:
        tx = random.uniform(-translation_max, translation_max) * width
        ty = random.uniform(-translation_max, translation_max) * height
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
    
    # Zoom aléatoire
    if zoom_range > 0:
        zoom = random.uniform(1 - zoom_range, 1 + zoom_range)
        cx, cy = width/2, height/2
        M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
    
    # Retournement horizontal aléatoire
    if flip_horizontal and random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Remettre la dimension du canal si nécessaire
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    return image

def charger_images_avec_etiquettes(dossier_normal, dossier_pneumonie, taille_cible=(150, 150), 
                                 normaliser=True, limite=None):
    """
    Charge et prétraite des images des deux classes avec leurs étiquettes.
    
    Args:
        dossier_normal (str): Chemin vers le dossier d'images normales.
        dossier_pneumonie (str): Chemin vers le dossier d'images avec pneumonie.
        taille_cible (tuple): Dimensions cibles des images.
        normaliser (bool): Si True, normalise les valeurs de pixels.
        limite (int): Nombre maximal d'images à charger par classe (None pour toutes).
        
    Returns:
        tuple: (X, y) où X est un tableau d'images et y un tableau d'étiquettes (0 pour normal, 1 pour pneumonie).
    """
    # Charger les images normales
    X_normal = pretraiter_lot_images(dossier_normal, taille_cible, normaliser, limite=limite)
    y_normal = np.zeros(len(X_normal))
    
    # Charger les images avec pneumonie
    X_pneumonie = pretraiter_lot_images(dossier_pneumonie, taille_cible, normaliser, limite=limite)
    y_pneumonie = np.ones(len(X_pneumonie))
    
    # Combiner les données
    X = np.vstack([X_normal, X_pneumonie])
    y = np.hstack([y_normal, y_pneumonie])
    
    return X, y

def creer_batch_avec_augmentation(X, y, batch_size=32, augmentation=True):
    """
    Crée un batch d'images avec augmentation de données.
    
    Args:
        X (numpy.ndarray): Tableau d'images.
        y (numpy.ndarray): Tableau d'étiquettes.
        batch_size (int): Taille du batch.
        augmentation (bool): Si True, applique l'augmentation de données.
        
    Returns:
        tuple: (X_batch, y_batch)
    """
    # Sélectionner des indices aléatoires
    indices = np.random.choice(len(X), batch_size, replace=False)
    
    # Créer le batch
    X_batch = X[indices].copy()
    y_batch = y[indices].copy()
    
    # Appliquer l'augmentation de données si nécessaire
    if augmentation:
        for i in range(len(X_batch)):
            X_batch[i] = augmenter_image(X_batch[i])
    
    return X_batch, y_batch

def afficher_exemples_augmentation(image, nb_exemples=5):
    """
    Affiche des exemples d'augmentation d'une image.
    
    Args:
        image (numpy.ndarray): Image à augmenter.
        nb_exemples (int): Nombre d'exemples à afficher.
    """
    # Afficher l'image originale et les exemples augmentés
    plt.figure(figsize=(15, 3))
    
    # Image originale
    plt.subplot(1, nb_exemples + 1, 1)
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Images augmentées
    for i in range(nb_exemples):
        aug_image = augmenter_image(image)
        plt.subplot(1, nb_exemples + 1, i + 2)
        plt.imshow(np.squeeze(aug_image), cmap='gray')
        plt.title(f'Aug #{i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def diviser_donnees(X, y, test_size=0.2, random_state=42):
    """
    Divise un ensemble de données en ensembles d'entraînement et de test.
    
    Args:
        X (numpy.ndarray): Données d'entrée.
        y (numpy.ndarray): Étiquettes.
        test_size (float): Proportion des données à utiliser pour le test.
        random_state (int): Graine aléatoire pour la reproductibilité.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def preparer_donnees_pour_modele(train_dir, test_dir, img_size=(150, 150), batch_size=32, validation_size=0.25, limite_train=None, limite_test=None, val_dir=None):
    """
    Prépare les données pour l'entraînement d'un modèle.
    Peut fusionner train+val puis refaire un split équilibré si val_dir est fourni.
    
    Args:
        train_dir (str): Chemin vers le dossier d'entraînement.
        test_dir (str): Chemin vers le dossier de test.
        img_size (tuple): Dimensions cibles des images (hauteur, largeur).
        batch_size (int): Taille des batchs.
        validation_size (float): Proportion des données d'entraînement à utiliser pour la validation.
        limite_train (int, optional): Limite le nombre d'images chargées pour l'entraînement.
        limite_test (int, optional): Limite le nombre d'images chargées pour le test.
        val_dir (str, optional): Chemin vers le dossier de validation (pour fusionner train+val).
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Chemins vers les catégories
    train_normal_dir = os.path.join(train_dir, 'NORMAL')
    train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')
    test_normal_dir = os.path.join(test_dir, 'NORMAL')
    test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')

    if val_dir is not None:
        # Fusionner train + val, puis refaire un split
        val_normal_dir = os.path.join(val_dir, 'NORMAL')
        val_pneumonia_dir = os.path.join(val_dir, 'PNEUMONIA')
        print(f"Chargement des images de train ({train_dir}) et val ({val_dir}) pour fusion...")
        X_train_raw, y_train_raw = charger_images_avec_etiquettes(train_normal_dir, train_pneumonia_dir, img_size, limite=limite_train)
        X_val_raw, y_val_raw = charger_images_avec_etiquettes(val_normal_dir, val_pneumonia_dir, img_size)
        X_all = np.concatenate([X_train_raw, X_val_raw])
        y_all = np.concatenate([y_train_raw, y_val_raw])
        print(f"Fusion train+val: {X_all.shape[0]} images")
        # Nouveau split stratifié
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=max(validation_size, 100/len(X_all)), random_state=42, stratify=y_all
        )
        print(f"\nNouveau split train/val:")
        print(f"Ensemble d'entraînement: {X_train.shape[0]} images")
        print(f"  - Images normales: {np.sum(y_train == 0)}")
        print(f"  - Images avec pneumonie: {np.sum(y_train == 1)}")
        print(f"Ensemble de validation: {X_val.shape[0]} images")
        print(f"  - Images normales: {np.sum(y_val == 0)}")
        print(f"  - Images avec pneumonie: {np.sum(y_val == 1)}")
    else:
        # Comportement classique
        X_train, y_train = charger_images_avec_etiquettes(train_normal_dir, train_pneumonia_dir, img_size, limite=limite_train)
        print(f"Données d'entraînement chargées: {X_train.shape[0]} images")
        print(f"  - Images normales: {np.sum(y_train == 0)}")
        print(f"  - Images avec pneumonie: {np.sum(y_train == 1)}")
        # Split train/val
        X_train, X_val, y_train, y_val = diviser_donnees(X_train, y_train, test_size=validation_size)
        print(f"\nRépartition après division:")
        print(f"Ensemble d'entraînement: {X_train.shape[0]} images")
        print(f"  - Images normales: {np.sum(y_train == 0)}")
        print(f"  - Images avec pneumonie: {np.sum(y_train == 1)}")
        print(f"Ensemble de validation: {X_val.shape[0]} images")
        print(f"  - Images normales: {np.sum(y_val == 0)}")
        print(f"  - Images avec pneumonie: {np.sum(y_val == 1)}")

    # Charger les données de test
    X_test, y_test = charger_images_avec_etiquettes(test_normal_dir, test_pneumonia_dir, img_size, limite=limite_test)
    print(f"\nEnsemble de test: {X_test.shape[0]} images")
    print(f"  - Images normales: {np.sum(y_test == 0)}")
    print(f"  - Images avec pneumonie: {np.sum(y_test == 1)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    # Exemple d'utilisation
    print("Ce module contient des fonctions pour le prétraitement d'images.")
    print("Importez-le dans votre script principal pour l'utiliser.")
