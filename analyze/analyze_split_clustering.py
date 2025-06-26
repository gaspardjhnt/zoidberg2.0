import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# Charger les données prétraitées (à adapter selon ton pipeline)
# On suppose que tu as déjà X_train, X_val, X_test, y_train, y_val, y_test sauvegardés ou accessibles
# Sinon, il faut les charger avec le même code que dans le notebook
from src.preprocessing.preprocess_images import preparer_donnees_pour_modele

data_dir = 'data/chest_Xray'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')
img_size = (150, 150)

X_train, y_train, X_val, y_val, X_test, y_test = preparer_donnees_pour_modele(
    train_dir, test_dir, img_size=img_size, validation_size=0.25, val_dir=val_dir
)

# Aplatir les images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Concaténer toutes les données
X_all = np.concatenate([X_train_flat, X_val_flat, X_test_flat])
y_all = np.concatenate([y_train, y_val, y_test])
splits = (['Train'] * len(y_train)) + (['Val'] * len(y_val)) + (['Test'] * len(y_test))
splits = np.array(splits)

# Réduire la dimensionnalité (PCA puis t-SNE pour la visualisation)
print('Réduction de dimensionnalité avec PCA...')
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_all)

print('Application de t-SNE pour visualisation 2D...')
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X_pca)

# Visualisation
plt.figure(figsize=(10, 8))
colors = {'Train': 'blue', 'Val': 'green', 'Test': 'red'}
for split in ['Train', 'Val', 'Test']:
    mask = splits == split
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[split], label=split, alpha=0.5, s=10)
plt.legend()
plt.title('t-SNE des images (couleur = split)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('split_tsne.png', dpi=300)
plt.show()

print('Analyse visuelle terminée. Regarde le fichier split_tsne.png pour voir si les points Test sont mélangés ou isolés.') 