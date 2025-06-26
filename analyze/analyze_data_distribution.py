import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import pandas as pd
import os

def analyze_data_distribution():
    """Analyse la distribution des données pour comprendre les différences validation/test"""
    
    print("🔍 ANALYSE DE LA DISTRIBUTION DES DONNÉES")
    print("=" * 60)
    
    # Charger les données
    try:
        # Ajouter le répertoire src au chemin Python
        import sys
        sys.path.append('src')
        
        from preprocessing.preprocess_images import charger_images_avec_etiquettes
        
        print("📊 Chargement des données...")
        
        # Définir les chemins
        base_dir = os.path.join('data', 'chest_Xray')
        train_normal = os.path.join(base_dir, 'train', 'NORMAL')
        train_pneumonia = os.path.join(base_dir, 'train', 'PNEUMONIA')
        val_normal = os.path.join(base_dir, 'val', 'NORMAL')
        val_pneumonia = os.path.join(base_dir, 'val', 'PNEUMONIA')
        test_normal = os.path.join(base_dir, 'test', 'NORMAL')
        test_pneumonia = os.path.join(base_dir, 'test', 'PNEUMONIA')
        
        # Charger les données
        X_train, y_train = charger_images_avec_etiquettes(train_normal, train_pneumonia)
        X_val, y_val = charger_images_avec_etiquettes(val_normal, val_pneumonia)
        X_test, y_test = charger_images_avec_etiquettes(test_normal, test_pneumonia)
        
        print(f"✅ Données chargées avec succès!")
        print(f"   Train: {X_train.shape} - {np.sum(y_train)} pneumonies sur {len(y_train)}")
        print(f"   Validation: {X_val.shape} - {np.sum(y_val)} pneumonies sur {len(y_val)}")
        print(f"   Test: {X_test.shape} - {np.sum(y_test)} pneumonies sur {len(y_test)}")
        
        # Aplatir les images pour l'analyse
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # 1. ANALYSE DE LA RÉPARTITION DES CLASSES
        print("\n📈 RÉPARTITION DES CLASSES:")
        print(f"   Train - Pneumonie: {np.mean(y_train)*100:.1f}% | Normal: {(1-np.mean(y_train))*100:.1f}%")
        print(f"   Validation - Pneumonie: {np.mean(y_val)*100:.1f}% | Normal: {(1-np.mean(y_val))*100:.1f}%")
        print(f"   Test - Pneumonie: {np.mean(y_test)*100:.1f}% | Normal: {(1-np.mean(y_test))*100:.1f}%")
        
        # 2. ANALYSE STATISTIQUE DES IMAGES
        print("\n📊 STATISTIQUES DES IMAGES:")
        for name, data, labels in [("Train", X_train_flat, y_train), ("Validation", X_val_flat, y_val), ("Test", X_test_flat, y_test)]:
            print(f"\n   {name}:")
            print(f"     Moyenne: {np.mean(data):.3f}")
            print(f"     Écart-type: {np.std(data):.3f}")
            print(f"     Min: {np.min(data):.3f}")
            print(f"     Max: {np.max(data):.3f}")
            
            # Par classe
            normal_data = data[labels == 0]
            pneumonie_data = data[labels == 1]
            print(f"     Normal - Moyenne: {np.mean(normal_data):.3f}, Écart-type: {np.std(normal_data):.3f}")
            print(f"     Pneumonie - Moyenne: {np.mean(pneumonie_data):.3f}, Écart-type: {np.std(pneumonie_data):.3f}")
        
        # 3. VISUALISATION DES DISTRIBUTIONS
        print("\n📊 VISUALISATION DES DISTRIBUTIONS...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Distribution des valeurs de pixels
        for i, (name, data, labels) in enumerate([("Train", X_train_flat, y_train), ("Validation", X_val_flat, y_val), ("Test", X_test_flat, y_test)]):
            axes[0, i].hist(data.flatten(), bins=50, alpha=0.7, label=name)
            axes[0, i].set_title(f'Distribution des pixels - {name}')
            axes[0, i].set_xlabel('Valeur de pixel')
            axes[0, i].set_ylabel('Fréquence')
            axes[0, i].legend()
        
        # Distribution par classe
        for i, (name, data, labels) in enumerate([("Train", X_train_flat, y_train), ("Validation", X_val_flat, y_val), ("Test", X_test_flat, y_test)]):
            normal_data = data[labels == 0]
            pneumonie_data = data[labels == 1]
            
            axes[1, i].hist(normal_data.flatten(), bins=30, alpha=0.7, label='Normal', color='blue')
            axes[1, i].hist(pneumonie_data.flatten(), bins=30, alpha=0.7, label='Pneumonie', color='red')
            axes[1, i].set_title(f'Distribution par classe - {name}')
            axes[1, i].set_xlabel('Valeur de pixel')
            axes[1, i].set_ylabel('Fréquence')
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. CLUSTERING POUR DÉTECTER DES GROUPES NATURELS
        print("\n🔬 ANALYSE PAR CLUSTERING...")
        
        # Réduire la dimensionnalité pour le clustering
        print("   Réduction de dimensionnalité avec PCA...")
        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(X_train_flat)
        X_val_pca = pca.transform(X_val_flat)
        X_test_pca = pca.transform(X_test_flat)
        
        # K-means clustering
        print("   Application de K-means clustering...")
        kmeans = KMeans(n_clusters=4, random_state=42)
        
        # Combiner toutes les données pour le clustering
        all_data = np.vstack([X_train_pca, X_val_pca, X_test_pca])
        all_labels = np.concatenate([y_train, y_val, y_test])
        all_sets = np.concatenate([['Train']*len(y_train), ['Validation']*len(y_val), ['Test']*len(y_test)])
        
        cluster_labels = kmeans.fit_predict(all_data)
        
        # Analyser la répartition des clusters
        print("\n📊 RÉPARTITION DES CLUSTERS:")
        for cluster_id in range(4):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = all_data[cluster_mask]
            cluster_labels_binary = all_labels[cluster_mask]
            cluster_sets = all_sets[cluster_mask]
            
            print(f"\n   Cluster {cluster_id}:")
            print(f"     Taille: {len(cluster_data)}")
            print(f"     % Pneumonie: {np.mean(cluster_labels_binary)*100:.1f}%")
            
            # Répartition par ensemble
            for set_name in ['Train', 'Validation', 'Test']:
                set_mask = cluster_sets == set_name
                if np.any(set_mask):
                    set_data = cluster_data[set_mask]
                    set_labels = cluster_labels_binary[set_mask]
                    print(f"     {set_name}: {len(set_data)} échantillons ({np.mean(set_labels)*100:.1f}% pneumonie)")
        
        # 5. VISUALISATION DES CLUSTERS
        print("\n📊 VISUALISATION DES CLUSTERS...")
        
        # t-SNE pour visualisation 2D
        print("   Application de t-SNE pour visualisation...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        all_data_2d = tsne.fit_transform(all_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Visualisation par cluster
        scatter1 = axes[0].scatter(all_data_2d[:, 0], all_data_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        axes[0].set_title('Clustering des données (K-means)')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Visualisation par ensemble (Train/Val/Test)
        colors = {'Train': 'blue', 'Validation': 'green', 'Test': 'red'}
        for set_name in ['Train', 'Validation', 'Test']:
            mask = all_sets == set_name
            axes[1].scatter(all_data_2d[mask, 0], all_data_2d[mask, 1], 
                          c=colors[set_name], label=set_name, alpha=0.6)
        
        axes[1].set_title('Répartition Train/Validation/Test')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. DÉTECTION D'ANOMALIES
        print("\n🚨 DÉTECTION D'ANOMALIES...")
        
        # Calculer les distances aux centroïdes
        distances = kmeans.transform(all_data)
        min_distances = np.min(distances, axis=1)
        
        # Identifier les anomalies (échantillons très éloignés des centroïdes)
        threshold = np.percentile(min_distances, 95)  # 5% des échantillons les plus éloignés
        anomalies = min_distances > threshold
        
        print(f"   Seuil d'anomalie (95e percentile): {threshold:.3f}")
        print(f"   Nombre d'anomalies détectées: {np.sum(anomalies)}")
        
        # Répartition des anomalies par ensemble
        for set_name in ['Train', 'Validation', 'Test']:
            mask = all_sets == set_name
            set_anomalies = anomalies[mask]
            print(f"   Anomalies dans {set_name}: {np.sum(set_anomalies)} ({np.mean(set_anomalies)*100:.1f}%)")
        
        # 7. ANALYSE DE LA SÉPARABILITÉ
        print("\n🔍 ANALYSE DE LA SÉPARABILITÉ DES ENSEMBLES...")
        
        # Calculer la distance moyenne entre les ensembles
        from scipy.spatial.distance import cdist
        
        train_mean = np.mean(X_train_pca, axis=0)
        val_mean = np.mean(X_val_pca, axis=0)
        test_mean = np.mean(X_test_pca, axis=0)
        
        train_val_dist = np.linalg.norm(train_mean - val_mean)
        train_test_dist = np.linalg.norm(train_mean - test_mean)
        val_test_dist = np.linalg.norm(val_mean - test_mean)
        
        print(f"   Distance Train-Validation: {train_val_dist:.3f}")
        print(f"   Distance Train-Test: {train_test_dist:.3f}")
        print(f"   Distance Validation-Test: {val_test_dist:.3f}")
        
        return {
            'train_data': X_train,
            'val_data': X_val,
            'test_data': X_test,
            'train_labels': y_train,
            'val_labels': y_val,
            'test_labels': y_test,
            'cluster_labels': cluster_labels,
            'anomalies': anomalies,
            'distances': {
                'train_val': train_val_dist,
                'train_test': train_test_dist,
                'val_test': val_test_dist
            }
        }
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = analyze_data_distribution() 