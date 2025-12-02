#!/usr/bin/env python3
"""
Vérifie et affiche la configuration MLflow
"""
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour pouvoir importer mlflow_config
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from mlflow_config import MLFLOW_CONFIG, setup_mlflow
import mlflow

def main():
    print("=" * 70)
    print("🔍 VÉRIFICATION CONFIGURATION MLFLOW")
    print("=" * 70)
    
    print(f"\n📁 Chemins:")
    for name, path in MLFLOW_CONFIG["PATHS"].items():
        print(f"  {name:20}: {path}")
        print(f"    {'✓ Existe' if path.exists() else '✗ Manquant'}")
    
    print(f"\n⚙️ Configuration MLflow:")
    print(f"  Nom expérience: '{MLFLOW_CONFIG['EXPERIMENT_NAME']}'")
    print(f"  Modèle enregistré: '{MLFLOW_CONFIG['REGISTERED_MODEL_NAME']}'")
    print(f"  URI fichier: {MLFLOW_CONFIG['TRACKING_URI_FILE']}")
    print(f"  URI HTTP: {MLFLOW_CONFIG['TRACKING_URI_HTTP']}")
    
    print(f"\n🏷️ Tags par défaut:")
    for tag, value in MLFLOW_CONFIG["DEFAULT_TAGS"].items():
        print(f"  {tag}: {value}")
    
    # Vérifier la connexion MLflow
    print(f"\n🔗 Test connexion MLflow:")
    try:
        # Mode entraînement
        mlflow_train = setup_mlflow('train')
        experiments = mlflow.search_experiments()
        print(f"  Mode 'train': ✓ OK")
        print(f"  Expériences trouvées: {len(experiments)}")
        
        # Vérifier si notre expérience existe
        exp_names = [exp.name for exp in experiments]
        target_exp = MLFLOW_CONFIG["EXPERIMENT_NAME"]
        if target_exp in exp_names:
            print(f"  ✓ Expérience '{target_exp}' trouvée")
            # Afficher les détails
            exp = mlflow.get_experiment_by_name(target_exp)
            print(f"    ID: {exp.experiment_id}")
            print(f"    Créée: {exp.creation_time}")
        else:
            print(f"  ⚠️  Expérience '{target_exp}' non trouvée")
            print(f"    (sera créée au premier run)")
        
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
    
    # Vérifier les modèles
    print(f"\n🤖 Vérification des modèles:")
    models_dir = MLFLOW_CONFIG["PATHS"]["models"]
    models = list(models_dir.glob("*.pkl"))
    if models:
        print(f"  Modèles trouvés: {len(models)}")
        for model in models[:3]:  # Afficher les 3 premiers
            print(f"    - {model.name}")
        if len(models) > 3:
            print(f"    ... et {len(models) - 3} autres")
    else:
        print(f"  ⚠️  Aucun modèle trouvé dans {models_dir}")
        print(f"    Lancez: python train.py baseline")
    
    print("\n" + "=" * 70)
    print("✅ Vérification terminée")
    print("=" * 70)
    
    print(f"\n📋 Résumé:")
    print(f"  Expérience unifiée: '{MLFLOW_CONFIG['EXPERIMENT_NAME']}'")
    print(f"  Train.py utilisera: URI fichier")
    print(f"  Predict.py utilisera: URI HTTP")
    print(f"  Modèle enregistré: '{MLFLOW_CONFIG['REGISTERED_MODEL_NAME']}'")

if __name__ == "__main__":
    main()