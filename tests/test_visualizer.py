"""
Tests unitaires pour le module de visualisation.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au chemin
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.visualizer import afficher_tableau_comparatif_modeles

class TestVisualizer(unittest.TestCase):
    """Tests pour les fonctions de visualisation."""

    @patch('matplotlib.pyplot.show')
    def test_afficher_tableau_comparatif_modeles(self, mock_show):
        """Test de la fonction afficher_tableau_comparatif_modeles."""
        # Données de test
        modeles = ["Modèle A", "Modèle B", "Modèle C"]
        metriques = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC', 'Temps (s)']
        valeurs = [
            [0.9582, 0.9623, 0.9541, 0.9582, 0.9823, 2.345],  # Modèle A
            [0.9721, 0.9812, 0.9634, 0.9721, 0.9901, 5.678],  # Modèle B
            [0.9321, 0.9412, 0.9234, 0.9321, 0.9512, 1.234]   # Modèle C
        ]
        
        # Appeler la fonction à tester
        afficher_tableau_comparatif_modeles(modeles, metriques, valeurs, "Test Tableau")
        
        # Vérifier que plt.show() a été appelé
        mock_show.assert_called_once()
        
    def test_formatage_pourcentage(self):
        """Test du formatage des pourcentages dans le tableau."""
        # Données de test avec une seule métrique (Accuracy) et un seul modèle
        modeles = ["Test Model"]
        metriques = ['Accuracy']
        valeurs = [[0.9582]]
        
        # Utiliser un mock pour capturer le DataFrame créé
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show'):
            mock_ax = MagicMock()
            mock_subplots.return_value = (MagicMock(), mock_ax)
            
            # Appeler la fonction
            afficher_tableau_comparatif_modeles(modeles, metriques, valeurs)
            
            # Vérifier que table() a été appelé avec les bonnes valeurs
            args, kwargs = mock_ax.table.call_args
            
            # Vérifier que le formatage en pourcentage a été appliqué
            # La valeur devrait être "95.82%" et non "0.9582"
            self.assertIn('cellText', kwargs)
            cell_text = kwargs['cellText']
            self.assertTrue(any("95.82%" in str(row) for row in cell_text))
            
    def test_formatage_temps(self):
        """Test du formatage des temps dans le tableau."""
        # Données de test avec une seule métrique (Temps) et un seul modèle
        modeles = ["Test Model"]
        metriques = ['Temps (s)']
        valeurs = [[2.3456789]]
        
        # Utiliser un mock pour capturer le DataFrame créé
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.show'):
            mock_ax = MagicMock()
            mock_subplots.return_value = (MagicMock(), mock_ax)
            
            # Appeler la fonction
            afficher_tableau_comparatif_modeles(modeles, metriques, valeurs)
            
            # Vérifier que table() a été appelé avec les bonnes valeurs
            args, kwargs = mock_ax.table.call_args
            
            # Vérifier que le formatage à 3 décimales a été appliqué
            # La valeur devrait être "2.346" et non "2.3456789"
            self.assertIn('cellText', kwargs)
            cell_text = kwargs['cellText']
            self.assertTrue(any("2.346" in str(row) for row in cell_text))

if __name__ == '__main__':
    unittest.main()
