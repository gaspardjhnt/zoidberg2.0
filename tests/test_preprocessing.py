"""
Tests supplémentaires pour les fonctions de prétraitement d'images
Ce script contient des tests additionnels pour les fonctions du module preprocess_images.py
À ajouter au notebook test_pretraitement_sans_tf.ipynb
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random

# Import de notre module de prétraitement
from src.preprocessing import preprocess_images as preproc

# Configuration pour afficher les images dans le notebook
plt.rcParams['figure.figsize'] = (10, 8)
plt.style.use('ggplot')

# Chemins définis dans le notebook
data_dir = r'data/chest_Xray'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
train_normal_dir = os.path.join(train_dir, 'NORMAL')
train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')
test_normal_dir = os.path.join(test_dir, 'NORMAL')
test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')

# Taille d'image standard pour les tests
img_size = (150, 150)

# -------------------------------------------------------------------------
# 6. Test de la fonction pretraiter_lot_images
# -------------------------------------------------------------------------

def test_pretraiter_lot_images():
    """
    Teste la fonction pretraiter_lot_images qui prétraite un lot d'images dans un dossier.
    """
    # Prétraiter un petit lot d'images normales
    print("Test de pretraiter_lot_images sur des images normales")
    images_normales = preproc.pretraiter_lot_images(train_normal_dir, img_size, limite=5)
    
    # Afficher les informations sur le lot d'images
    print(f"Forme du lot d'images: {images_normales.shape}")
    print(f"Valeur minimale: {images_normales.min()}")
    print(f"Valeur maximale: {images_normales.max()}")
    print(f"Valeur moyenne: {images_normales.mean():.4f}")
    
    # Afficher les images prétraitées
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(np.squeeze(images_normales[i]), cmap='gray')
        ax.set_title(f"Image {i+1}")
        ax.axis('off')
    plt.suptitle("Lot d'images normales prétraitées", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Prétraiter un petit lot d'images avec pneumonie
    print("\nTest de pretraiter_lot_images sur des images avec pneumonie")
    images_pneumonie = preproc.pretraiter_lot_images(train_pneumonia_dir, img_size, limite=5)
    
    # Afficher les informations sur le lot d'images
    print(f"Forme du lot d'images: {images_pneumonie.shape}")
    print(f"Valeur minimale: {images_pneumonie.min()}")
    print(f"Valeur maximale: {images_pneumonie.max()}")
    print(f"Valeur moyenne: {images_pneumonie.mean():.4f}")
    
    # Afficher les images prétraitées
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(np.squeeze(images_pneumonie[i]), cmap='gray')
        ax.set_title(f"Image {i+1}")
        ax.axis('off')
    plt.suptitle("Lot d'images avec pneumonie prétraitées", fontsize=16)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# 7. Test de la fonction diviser_donnees
# -------------------------------------------------------------------------

def test_diviser_donnees():
    """
    Teste la fonction diviser_donnees qui divise un ensemble de données en ensembles d'entraînement et de test.
    """
    # Charger un petit lot d'images pour le test
    print("Test de diviser_donnees")
    X, y = preproc.charger_images_avec_etiquettes(train_normal_dir, train_pneumonia_dir, img_size, limite=20)
    
    # Diviser les données
    X_train, X_test, y_train, y_test = preproc.diviser_donnees(X, y, test_size=0.25, random_state=42)
    
    # Afficher les informations sur les ensembles de données
    print(f"Forme de X: {X.shape}")
    print(f"Forme de y: {y.shape}")
    print(f"Forme de X_train: {X_train.shape}")
    print(f"Forme de X_test: {X_test.shape}")
    print(f"Forme de y_train: {y_train.shape}")
    print(f"Forme de y_test: {y_test.shape}")
    print(f"Nombre d'images normales dans l'ensemble d'entraînement: {np.sum(y_train == 0)}")
    print(f"Nombre d'images avec pneumonie dans l'ensemble d'entraînement: {np.sum(y_train == 1)}")
    print(f"Nombre d'images normales dans l'ensemble de test: {np.sum(y_test == 0)}")
    print(f"Nombre d'images avec pneumonie dans l'ensemble de test: {np.sum(y_test == 1)}")
    
    # Visualiser quelques images de l'ensemble d'entraînement
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    axes = axes.flatten()
    
    # Images normales de l'ensemble d'entraînement
    normal_indices = np.where(y_train == 0)[0][:4]
    for i, idx in enumerate(normal_indices):
        axes[i].imshow(np.squeeze(X_train[idx]), cmap='gray')
        axes[i].set_title(f"Normal (train)")
        axes[i].axis('off')
    
    # Images avec pneumonie de l'ensemble d'entraînement
    pneumonia_indices = np.where(y_train == 1)[0][:4]
    for i, idx in enumerate(pneumonia_indices):
        axes[i+4].imshow(np.squeeze(X_train[idx]), cmap='gray')
        axes[i+4].set_title(f"Pneumonie (train)")
        axes[i+4].axis('off')
    
    plt.suptitle("Échantillons de l'ensemble d'entraînement", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Visualiser quelques images de l'ensemble de test
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    axes = axes.flatten()
    
    # Images normales de l'ensemble de test
    normal_indices = np.where(y_test == 0)[0][:4]
    for i, idx in enumerate(normal_indices):
        axes[i].imshow(np.squeeze(X_test[idx]), cmap='gray')
        axes[i].set_title(f"Normal (test)")
        axes[i].axis('off')
    
    # Images avec pneumonie de l'ensemble de test
    pneumonia_indices = np.where(y_test == 1)[0][:4]
    for i, idx in enumerate(pneumonia_indices):
        axes[i+4].imshow(np.squeeze(X_test[idx]), cmap='gray')
        axes[i+4].set_title(f"Pneumonie (test)")
        axes[i+4].axis('off')
    
    plt.suptitle("Échantillons de l'ensemble de test", fontsize=16)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# 8. Analyse des statistiques des images avant et après prétraitement
# -------------------------------------------------------------------------

def analyser_statistiques_images():
    """
    Analyse les statistiques des images avant et après prétraitement.
    """
    print("Analyse des statistiques des images avant et après prétraitement")
    
    # Sélectionner quelques images aléatoires
    normal_images = os.listdir(train_normal_dir)
    pneumonia_images = os.listdir(train_pneumonia_dir)
    
    selected_normal = random.sample(normal_images, 5)
    selected_pneumonia = random.sample(pneumonia_images, 5)
    
    # Créer des listes pour stocker les statistiques
    stats_original = []
    stats_processed = []
    
    # Analyser les images normales
    for img_name in selected_normal:
        img_path = os.path.join(train_normal_dir, img_name)
        
        # Image originale
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        # Statistiques de l'image originale
        orig_min = original_img.min()
        orig_max = original_img.max()
        orig_mean = original_img.mean()
        orig_std = original_img.std()
        
        stats_original.append({
            'type': 'Normal',
            'name': img_name,
            'min': orig_min,
            'max': orig_max,
            'mean': orig_mean,
            'std': orig_std
        })
        
        # Image prétraitée
        processed_img = preproc.charger_et_pretraiter_image(img_path, img_size)
        
        # Statistiques de l'image prétraitée
        proc_min = processed_img.min()
        proc_max = processed_img.max()
        proc_mean = processed_img.mean()
        proc_std = processed_img.std()
        
        stats_processed.append({
            'type': 'Normal',
            'name': img_name,
            'min': proc_min,
            'max': proc_max,
            'mean': proc_mean,
            'std': proc_std
        })
    
    # Analyser les images avec pneumonie
    for img_name in selected_pneumonia:
        img_path = os.path.join(train_pneumonia_dir, img_name)
        
        # Image originale
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        # Statistiques de l'image originale
        orig_min = original_img.min()
        orig_max = original_img.max()
        orig_mean = original_img.mean()
        orig_std = original_img.std()
        
        stats_original.append({
            'type': 'Pneumonie',
            'name': img_name,
            'min': orig_min,
            'max': orig_max,
            'mean': orig_mean,
            'std': orig_std
        })
        
        # Image prétraitée
        processed_img = preproc.charger_et_pretraiter_image(img_path, img_size)
        
        # Statistiques de l'image prétraitée
        proc_min = processed_img.min()
        proc_max = processed_img.max()
        proc_mean = processed_img.mean()
        proc_std = processed_img.std()
        
        stats_processed.append({
            'type': 'Pneumonie',
            'name': img_name,
            'min': proc_min,
            'max': proc_max,
            'mean': proc_mean,
            'std': proc_std
        })
    
    # Afficher les statistiques sous forme de tableau
    print("\nStatistiques des images originales:")
    print("Type\t\tMin\tMax\tMoyenne\tÉcart-type")
    print("-" * 50)
    for stat in stats_original:
        print(f"{stat['type']}\t{stat['min']:.1f}\t{stat['max']:.1f}\t{stat['mean']:.1f}\t{stat['std']:.1f}")
    
    print("\nStatistiques des images prétraitées:")
    print("Type\t\tMin\tMax\tMoyenne\tÉcart-type")
    print("-" * 50)
    for stat in stats_processed:
        print(f"{stat['type']}\t{stat['min']:.4f}\t{stat['max']:.4f}\t{stat['mean']:.4f}\t{stat['std']:.4f}")
    
    # Visualiser les distributions des valeurs de pixels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogrammes des images originales
    normal_vals = [s['mean'] for s in stats_original if s['type'] == 'Normal']
    pneumonia_vals = [s['mean'] for s in stats_original if s['type'] == 'Pneumonie']
    
    axes[0, 0].hist(normal_vals, bins=5, alpha=0.7, label='Normal')
    axes[0, 0].hist(pneumonia_vals, bins=5, alpha=0.7, label='Pneumonie')
    axes[0, 0].set_title('Distribution des moyennes (images originales)')
    axes[0, 0].set_xlabel('Valeur moyenne des pixels')
    axes[0, 0].set_ylabel('Nombre d\'images')
    axes[0, 0].legend()
    
    # Histogrammes des images prétraitées
    normal_vals = [s['mean'] for s in stats_processed if s['type'] == 'Normal']
    pneumonia_vals = [s['mean'] for s in stats_processed if s['type'] == 'Pneumonie']
    
    axes[0, 1].hist(normal_vals, bins=5, alpha=0.7, label='Normal')
    axes[0, 1].hist(pneumonia_vals, bins=5, alpha=0.7, label='Pneumonie')
    axes[0, 1].set_title('Distribution des moyennes (images prétraitées)')
    axes[0, 1].set_xlabel('Valeur moyenne des pixels')
    axes[0, 1].set_ylabel('Nombre d\'images')
    axes[0, 1].legend()
    
    # Écart-type des images originales
    normal_vals = [s['std'] for s in stats_original if s['type'] == 'Normal']
    pneumonia_vals = [s['std'] for s in stats_original if s['type'] == 'Pneumonie']
    
    axes[1, 0].hist(normal_vals, bins=5, alpha=0.7, label='Normal')
    axes[1, 0].hist(pneumonia_vals, bins=5, alpha=0.7, label='Pneumonie')
    axes[1, 0].set_title('Distribution des écarts-types (images originales)')
    axes[1, 0].set_xlabel('Écart-type des pixels')
    axes[1, 0].set_ylabel('Nombre d\'images')
    axes[1, 0].legend()
    
    # Écart-type des images prétraitées
    normal_vals = [s['std'] for s in stats_processed if s['type'] == 'Normal']
    pneumonia_vals = [s['std'] for s in stats_processed if s['type'] == 'Pneumonie']
    
    axes[1, 1].hist(normal_vals, bins=5, alpha=0.7, label='Normal')
    axes[1, 1].hist(pneumonia_vals, bins=5, alpha=0.7, label='Pneumonie')
    axes[1, 1].set_title('Distribution des écarts-types (images prétraitées)')
    axes[1, 1].set_xlabel('Écart-type des pixels')
    axes[1, 1].set_ylabel('Nombre d\'images')
    axes[1, 1].legend()
    
    plt.suptitle('Comparaison des statistiques avant et après prétraitement', fontsize=16)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# 9. Test de la fonction preparer_donnees_pour_modele
# -------------------------------------------------------------------------

def test_preparer_donnees_pour_modele():
    """
    Teste la fonction preparer_donnees_pour_modele qui prépare les données pour l'entraînement d'un modèle.
    """
    print("Test de preparer_donnees_pour_modele")
    
    # Préparer les données pour le modèle (avec un petit nombre d'images pour le test)
    X_train, y_train, X_val, y_val, X_test, y_test = preproc.preparer_donnees_pour_modele(
        train_dir, test_dir, img_size=(150, 150), batch_size=32, limite_train=20, limite_test=10
    )
    
    # Afficher les informations sur les ensembles de données
    print(f"Forme de X_train: {X_train.shape}")
    print(f"Forme de y_train: {y_train.shape}")
    print(f"Forme de X_val: {X_val.shape}")
    print(f"Forme de y_val: {y_val.shape}")
    print(f"Forme de X_test: {X_test.shape}")
    print(f"Forme de y_test: {y_test.shape}")
    
    print(f"Nombre d'images normales dans l'ensemble d'entraînement: {np.sum(y_train == 0)}")
    print(f"Nombre d'images avec pneumonie dans l'ensemble d'entraînement: {np.sum(y_train == 1)}")
    print(f"Nombre d'images normales dans l'ensemble de validation: {np.sum(y_val == 0)}")
    print(f"Nombre d'images avec pneumonie dans l'ensemble de validation: {np.sum(y_val == 1)}")
    print(f"Nombre d'images normales dans l'ensemble de test: {np.sum(y_test == 0)}")
    print(f"Nombre d'images avec pneumonie dans l'ensemble de test: {np.sum(y_test == 1)}")
    
    # Visualiser quelques images de chaque ensemble
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    
    # Images de l'ensemble d'entraînement
    for i in range(4):
        idx = np.random.randint(0, len(X_train))
        axes[0, i].imshow(np.squeeze(X_train[idx]), cmap='gray')
        axes[0, i].set_title(f"{'Normal' if y_train[idx] == 0 else 'Pneumonie'} (train)")
        axes[0, i].axis('off')
    
    # Images de l'ensemble de validation
    for i in range(4):
        idx = np.random.randint(0, len(X_val))
        axes[1, i].imshow(np.squeeze(X_val[idx]), cmap='gray')
        axes[1, i].set_title(f"{'Normal' if y_val[idx] == 0 else 'Pneumonie'} (val)")
        axes[1, i].axis('off')
    
    # Images de l'ensemble de test
    for i in range(4):
        idx = np.random.randint(0, len(X_test))
        axes[2, i].imshow(np.squeeze(X_test[idx]), cmap='gray')
        axes[2, i].set_title(f"{'Normal' if y_test[idx] == 0 else 'Pneumonie'} (test)")
        axes[2, i].axis('off')
    
    plt.suptitle("Échantillons des ensembles d'entraînement, validation et test", fontsize=16)
    plt.tight_layout()
    plt.show()

# Fonction principale pour exécuter tous les tests
def executer_tous_les_tests():
    test_pretraiter_lot_images()
    test_diviser_donnees()
    analyser_statistiques_images()
    test_preparer_donnees_pour_modele()

if __name__ == "__main__":
    executer_tous_les_tests()
