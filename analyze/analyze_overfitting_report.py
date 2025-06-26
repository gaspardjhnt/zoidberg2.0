import numpy as np
from src.preprocessing.preprocess_images import preparer_donnees_pour_modele
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

base_dir = 'data/chest_Xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
img_size = (150, 150)

X_train, y_train, X_val, y_val, X_test, y_test = preparer_donnees_pour_modele(
    train_dir, test_dir, img_size=img_size, validation_size=0.25, val_dir=val_dir
)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_flat, y_train)

for split, X, y in [('Train', X_train_flat, y_train), ('Val', X_val_flat, y_val), ('Test', X_test_flat, y_test)]:
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"{split} - Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}") 