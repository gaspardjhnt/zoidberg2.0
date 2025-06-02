import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def afficher_tableau_comparatif_modeles(modeles, metriques, valeurs, titre="Performances des modèles"):
    """
    Affiche un tableau comparatif des performances de différents modèles.
    
    Args:
        modeles (list): Liste des noms des modèles.
        metriques (list): Liste des noms des métriques (colonnes).
        valeurs (list): Liste de listes contenant les valeurs pour chaque modèle et métrique.
                       Format: [[valeurs_modele1], [valeurs_modele2], ...].
        titre (str): Titre du tableau.
    """
    # Créer un DataFrame pandas pour faciliter la création du tableau
    data = {}
    for i, metrique in enumerate(metriques):
        data[metrique] = [valeurs[j][i] for j in range(len(modeles))]
    
    df = pd.DataFrame(data, index=modeles)
    
    # Définir la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, len(modeles) * 0.8 + 2))
    
    # Masquer les axes
    ax.axis('tight')
    ax.axis('off')
    
    # Créer le tableau
    table = ax.table(cellText=df.values,
                    rowLabels=df.index,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Personnaliser l'apparence du tableau
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Appliquer des couleurs aux cellules (bleu clair pour l'en-tête)
    header_color = '#4993D2'
    row_colors = ['#f1f1f2', 'w']
    
    # Colorier les cellules d'en-tête
    for j, col in enumerate(df.columns):
        table[(0, j)].set_facecolor(header_color)
        table[(0, j)].set_text_props(color='white')
    
    # Colorier les cellules des noms de modèles
    for i, row in enumerate(df.index):
        cell = table[(i+1, -1)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white')
    
    # Colorier les lignes alternées
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            table[(i+1, j)].set_facecolor(row_colors[i % 2])
    
    # Ajouter un titre
    plt.title(titre, fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.show()