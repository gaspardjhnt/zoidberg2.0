"""
Tests unitaires pour le module de prédiction.
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au chemin
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predict import load_model, predict_pneumonia, batch_predict

class TestPrediction(unittest.TestCase):
    """Tests pour les fonctions de prédiction."""

    @patch('src.models.predict.load_model')
    @patch('src.models.predict.load_and_preprocess_image')
    def test_predict_pneumonia(self, mock_preprocess, mock_load_model):
        """Test de la fonction predict_pneumonia."""
        # Configurer les mocks
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_load_model.return_value = mock_model
        
        mock_preprocess.return_value = np.array([1, 2, 3, 4])
        
        # Appeler la fonction à tester
        result, probability = predict_pneumonia("fake_image.jpg")
        
        # Vérifier les résultats
        self.assertEqual(result, "Pneumonie")
        self.assertEqual(probability, 0.8)
        
        # Vérifier que les mocks ont été appelés correctement
        mock_load_model.assert_called_once()
        mock_preprocess.assert_called_once_with("fake_image.jpg")
        mock_model.predict_proba.assert_called_once()

    @patch('src.models.predict.load_model')
    @patch('src.models.predict.load_and_preprocess_image')
    def test_predict_pneumonia_normal(self, mock_preprocess, mock_load_model):
        """Test de la fonction predict_pneumonia pour un cas normal."""
        # Configurer les mocks
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        mock_load_model.return_value = mock_model
        
        mock_preprocess.return_value = np.array([1, 2, 3, 4])
        
        # Appeler la fonction à tester
        result, probability = predict_pneumonia("fake_image.jpg")
        
        # Vérifier les résultats
        self.assertEqual(result, "Normal")
        self.assertEqual(probability, 0.7)

    @patch('os.path.isfile')
    def test_load_model_file_not_found(self, mock_isfile):
        """Test du chargement du modèle quand le fichier n'existe pas."""
        mock_isfile.return_value = False
        
        # Appeler la fonction à tester
        model = load_model("nonexistent_model.pkl")
        
        # Vérifier le résultat
        self.assertIsNone(model)

    @patch('src.models.predict.predict_pneumonia')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_batch_predict(self, mock_listdir, mock_isdir, mock_predict):
        """Test de la fonction batch_predict."""
        # Configurer les mocks
        mock_isdir.return_value = True
        mock_listdir.return_value = ["img1.jpg", "img2.jpg", "not_an_image.txt"]
        
        # Configurer le mock de predict_pneumonia
        mock_predict.side_effect = [
            ("Pneumonie", 0.9),
            ("Normal", 0.8),
        ]
        
        # Créer un mock pour le modèle
        mock_model = MagicMock()
        
        # Appeler la fonction à tester
        results = batch_predict("fake_folder", model=mock_model)
        
        # Vérifier les résultats
        self.assertEqual(len(results), 2)
        self.assertEqual(results["img1.jpg"]["prediction"], "Pneumonie")
        self.assertEqual(results["img1.jpg"]["probability"], 0.9)
        self.assertEqual(results["img2.jpg"]["prediction"], "Normal")
        self.assertEqual(results["img2.jpg"]["probability"], 0.8)
        
        # Vérifier que predict_pneumonia a été appelé le bon nombre de fois
        self.assertEqual(mock_predict.call_count, 2)

if __name__ == '__main__':
    unittest.main()
