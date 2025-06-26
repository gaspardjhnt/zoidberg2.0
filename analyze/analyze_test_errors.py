import numpy as np
import matplotlib.pyplot as plt
import os
from src.preprocessing.preprocess_images import preparer_donnees_pour_modele
from sklearn.linear_model import LogisticRegression

# Charger les données
base_dir = 'data/chest_Xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
img_size = (150, 150)

X_train, y_train, X_val, y_val, X_test, y_test = preparer_donnees_pour_modele(
    train_dir, test_dir, img_size=img_size, validation_size=0.25, val_dir=val_dir
)

# Redimensionner
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Entraîner un modèle simple (régression logistique)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_flat, y_train)
y_test_pred = clf.predict(X_test_flat)

# Trouver les indices des erreurs
fp_idx = np.where((y_test_pred == 1) & (y_test == 0))[0]  # Faux positifs
fn_idx = np.where((y_test_pred == 0) & (y_test == 1))[0]  # Faux négatifs

# Afficher quelques exemples
n = 8
plt.figure(figsize=(16, 4))
for i, idx in enumerate(fp_idx[:n]):
    plt.subplot(2, n, i+1)
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    plt.title('Faux Positif')
    plt.axis('off')
for i, idx in enumerate(fn_idx[:n]):
    plt.subplot(2, n, n+i+1)
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    plt.title('Faux Négatif')
    plt.axis('off')
plt.tight_layout()
plt.savefig('test_errors.png', dpi=300)
plt.show()
print('Regarde test_errors.png pour voir les erreurs du test.') 