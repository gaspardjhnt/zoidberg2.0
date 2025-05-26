import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocessing.preprocess_images import charger_et_pretraiter_image as load_and_preprocess_image

def load_model(model_path='../../models/régression_logistique_model.pkl'):
    """Charge le modèle entraîné."""
    try:
        # Vérifier si le fichier existe
        if not os.path.isfile(model_path):
            print(f"Le fichier modèle {model_path} n'existe pas")
            return None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Modèle chargé avec succès depuis {model_path}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None

def predict_pneumonia(image_path, model=None):
    """Prédit si une radiographie thoracique présente des signes de pneumonie."""
    if model is None:
        model = load_model()
        if model is None:
            return None, None
    
    try:
        # Prétraiter l'image
        processed_image = load_and_preprocess_image(image_path)
        
        # Faire la prédiction
        prediction = model.predict([processed_image])[0]
        proba = model.predict_proba([processed_image])[0]
        
        result = "Pneumonie" if prediction == 1 else "Normal"
        probability = proba[1] if prediction == 1 else proba[0]
        
        return result, probability
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return None, None

def batch_predict(image_folder, model=None):
    """Effectue des prédictions sur un dossier d'images."""
    if model is None:
        model = load_model()
        if model is None:
            return None
    
    results = {}
    
    try:
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                result, probability = predict_pneumonia(image_path, model)
                results[filename] = {
                    "prediction": result,
                    "probability": probability
                }
        return results
    except Exception as e:
        print(f"Erreur lors du traitement par lots: {e}")
        return None

if __name__ == "__main__":
    # Exemple d'utilisation du script
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result, probability = predict_pneumonia(image_path)
        if result is not None:
            print(f"Prédiction: {result}")
            print(f"Probabilité: {probability:.4f}")
    else:
        print("Usage: python predict.py <chemin_vers_image>")
        print("ou: python predict.py --batch <chemin_vers_dossier>")
