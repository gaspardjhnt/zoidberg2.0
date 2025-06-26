import json
import re

def analyze_notebook(filepath):
    """Analyse complète d'un notebook Jupyter"""
    
    print("=" * 80)
    print("ANALYSE COMPLÈTE DU NOTEBOOK")
    print("=" * 80)
    
    # Lire le notebook
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Nombre total de cellules: {len(data['cells'])}")
    
    # Séparer les cellules par type
    markdown_cells = [cell for cell in data['cells'] if cell['cell_type'] == 'markdown']
    code_cells = [cell for cell in data['cells'] if cell['cell_type'] == 'code']
    
    print(f"📝 Cellules markdown: {len(markdown_cells)}")
    print(f"💻 Cellules code: {len(code_cells)}")
    
    # Analyser les cellules markdown
    print("\n" + "=" * 80)
    print("📝 CONTENU DES CELLULES MARKDOWN")
    print("=" * 80)
    
    for i, cell in enumerate(markdown_cells):
        source = ''.join(cell['source'])
        print(f"\n--- Cellule Markdown {i+1} ---")
        print(source)
    
    # Analyser les cellules de code
    print("\n" + "=" * 80)
    print("💻 CONTENU DES CELLULES DE CODE")
    print("=" * 80)
    
    for i, cell in enumerate(code_cells):
        source = ''.join(cell['source'])
        print(f"\n--- Cellule Code {i+1} ---")
        print(source)
        
        # Analyser les sorties si elles existent
        if 'outputs' in cell and cell['outputs']:
            print(f"\n📤 Sorties de la cellule {i+1}:")
            for j, output in enumerate(cell['outputs']):
                if output['output_type'] == 'stream':
                    print(f"  Sortie {j+1} (stream): {output['text']}")
                elif output['output_type'] == 'execute_result':
                    if 'data' in output and 'text/plain' in output['data']:
                        print(f"  Sortie {j+1} (result): {output['data']['text/plain']}")
                elif output['output_type'] == 'display_data':
                    if 'data' in output and 'text/plain' in output['data']:
                        print(f"  Sortie {j+1} (display): {output['data']['text/plain']}")
    
    # Extraire les informations importantes
    print("\n" + "=" * 80)
    print("🔍 ANALYSE DES ÉLÉMENTS CLÉS")
    print("=" * 80)
    
    # Chercher les modèles utilisés
    all_code = '\n'.join([''.join(cell['source']) for cell in code_cells])
    
    models_found = []
    if 'LogisticRegression' in all_code:
        models_found.append('LogisticRegression')
    if 'DecisionTreeClassifier' in all_code:
        models_found.append('DecisionTreeClassifier')
    if 'RandomForestClassifier' in all_code:
        models_found.append('RandomForestClassifier')
    if 'SVC' in all_code:
        models_found.append('SVC')
    if 'GradientBoostingClassifier' in all_code:
        models_found.append('GradientBoostingClassifier')
    
    print(f"🤖 Modèles utilisés: {', '.join(models_found)}")
    
    # Chercher les métriques d'évaluation
    metrics_found = []
    if 'accuracy_score' in all_code:
        metrics_found.append('Accuracy')
    if 'precision_score' in all_code:
        metrics_found.append('Precision')
    if 'recall_score' in all_code:
        metrics_found.append('Recall')
    if 'f1_score' in all_code:
        metrics_found.append('F1-Score')
    if 'confusion_matrix' in all_code:
        metrics_found.append('Confusion Matrix')
    if 'roc_curve' in all_code:
        metrics_found.append('ROC Curve')
    
    print(f"📈 Métriques d'évaluation: {', '.join(metrics_found)}")
    
    # Chercher les configurations PCA
    pca_configs = re.findall(r'PCA\(n_components=(\d+)\)', all_code)
    if pca_configs:
        print(f"🔧 Configurations PCA trouvées: {', '.join(pca_configs)} composantes")
    
    # Chercher les résultats numériques
    print("\n📊 RÉSULTATS NUMÉRIQUES TROUVÉS:")
    accuracy_scores = re.findall(r'accuracy[:\s]*([0-9.]+)', all_code, re.IGNORECASE)
    if accuracy_scores:
        print(f"  Accuracy scores: {accuracy_scores}")
    
    f1_scores = re.findall(r'f1[:\s]*([0-9.]+)', all_code, re.IGNORECASE)
    if f1_scores:
        print(f"  F1 scores: {f1_scores}")
    
    precision_scores = re.findall(r'precision[:\s]*([0-9.]+)', all_code, re.IGNORECASE)
    if precision_scores:
        print(f"  Precision scores: {precision_scores}")
    
    recall_scores = re.findall(r'recall[:\s]*([0-9.]+)', all_code, re.IGNORECASE)
    if recall_scores:
        print(f"  Recall scores: {recall_scores}")

if __name__ == "__main__":
    analyze_notebook('notebooks/05_model_training_sklearn_different_scoring-Copy1.ipynb') 