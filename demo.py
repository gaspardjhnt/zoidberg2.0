"""
Script de démonstration pour le projet de détection de pneumonie.
"""

import os
import sys
from src.models.predict import predict_pneumonia, batch_predict

def run_single_prediction(image_path):
    """Démontre la prédiction sur une seule image."""
    print(f"\n=== Prédiction pour l'image: {os.path.basename(image_path)} ===")
    result, probability = predict_pneumonia(image_path)
    
    if result is not None:
        print(f"Résultat: {result}")
        print(f"Probabilité: {probability:.4f} ({probability*100:.2f}%)")
        
        # Interprétation du résultat
        if result == "Pneumonie":
            confidence = "élevée" if probability > 0.9 else "modérée"
            print(f"Interprétation: Signes de pneumonie détectés avec une confiance {confidence}.")
        else:
            confidence = "élevée" if probability > 0.9 else "modérée"
            print(f"Interprétation: Aucun signe de pneumonie détecté avec une confiance {confidence}.")
    else:
        print("Erreur: Impossible de faire une prédiction pour cette image.")

def run_batch_prediction(folder_path):
    """Démontre la prédiction sur un dossier d'images."""
    print(f"\n=== Prédictions pour le dossier: {folder_path} ===")
    results = batch_predict(folder_path)
    
    if results:
        print(f"Nombre d'images analysées: {len(results)}")
        
        # Compter les résultats
        pneumonia_count = sum(1 for r in results.values() if r["prediction"] == "Pneumonie")
        normal_count = sum(1 for r in results.values() if r["prediction"] == "Normal")
        
        print(f"Images avec pneumonie détectée: {pneumonia_count}")
        print(f"Images normales: {normal_count}")
        
        # Afficher les résultats détaillés
        print("\nRésultats détaillés:")
        for filename, result in results.items():
            print(f"- {filename}: {result['prediction']} (Probabilité: {result['probability']:.4f})")
    else:
        print("Erreur: Impossible de traiter les images dans ce dossier.")

if __name__ == "__main__":
    print("=== DÉMONSTRATEUR DE DÉTECTION DE PNEUMONIE ===")
    
    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("\nUtilisation:")
        print("  Pour une seule image: python demo.py chemin/vers/image.jpg")
        print("  Pour un dossier d'images: python demo.py --batch chemin/vers/dossier")
        sys.exit(1)
    
    # Traiter les arguments
    if sys.argv[1] == "--batch" and len(sys.argv) > 2:
        folder_path = sys.argv[2]
        if os.path.isdir(folder_path):
            run_batch_prediction(folder_path)
        else:
            print(f"Erreur: Le chemin '{folder_path}' n'est pas un dossier valide.")
    else:
        image_path = sys.argv[1]
        if os.path.isfile(image_path):
            run_single_prediction(image_path)
        else:
            print(f"Erreur: Le fichier '{image_path}' n'existe pas.")
