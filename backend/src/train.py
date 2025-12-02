import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os
from pathlib import Path

warnings.filterwarnings('ignore')

from data_utils import prepare_features_for_training, scale_features

# Détection automatique de l'environnement
if os.path.exists('/app'):  # Dans Docker
    PROJECT_ROOT = Path('/app')
else:  # En local
    PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "dataSet"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Créer les dossiers s'ils n'existent pas
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_configs():
    """Différentes configurations de modèles à tester"""
    configs = {
        'baseline': {
            'colsample_bytree': [0.7],
            'learning_rate': [0.1],
            'max_depth': [9],
            'min_child_weight': [1],
            'n_estimators': [350],
            'subsample': [0.8],
            'gamma': [0],
            'reg_alpha': [0.1],
            'reg_lambda': [1.5]
        },
        'fast_model': {
            'colsample_bytree': [0.8],
            'learning_rate': [0.15],
            'max_depth': [6],
            'min_child_weight': [1],
            'n_estimators': [150],
            'subsample': [0.8],
            'gamma': [0],
            'reg_alpha': [0.05],
            'reg_lambda': [1.0]
        },
        'accurate_model': {
            'colsample_bytree': [0.6],
            'learning_rate': [0.05],
            'max_depth': [12],
            'min_child_weight': [1],
            'n_estimators': [500],
            'subsample': [0.9],
            'gamma': [0.1],
            'reg_alpha': [0.2],
            'reg_lambda': [2.0]
        },
        'lightweight_model': {
            'colsample_bytree': [0.9],
            'learning_rate': [0.2],
            'max_depth': [4],
            'min_child_weight': [5],
            'n_estimators': [100],
            'subsample': [0.7],
            'gamma': [0],
            'reg_alpha': [0],
            'reg_lambda': [1.0]
        }
    }
    return configs


# Configuration MLflow
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
mlflow.set_experiment("car_price_prediction")



def train_model(
    model_version='baseline',
    cleaned_csv_path=None,  # ← Changé de string hardcodé à None
    model_output_path=None
):
    """
    Entraîne le modèle XGBoost et sauvegarde le pipeline complet
    
    Args:
        model_version (str): Version du modèle
        cleaned_csv_path (str|Path): Chemin vers les données (auto si None)
        model_output_path (str|Path): Chemin de sortie (auto si None)
    """
    
    # Utiliser les chemins dynamiques
    if cleaned_csv_path is None:
        cleaned_csv_path = DATA_DIR / "cleaned_cardata3.csv"  # ← Utilise DATA_DIR
    else:
        cleaned_csv_path = Path(cleaned_csv_path)
    
    # Vérifier que le fichier existe
    if not cleaned_csv_path.exists():
        # Essayer un nom alternatif
        alt_path = DATA_DIR / "cleaned_cardata2.csv"
        if alt_path.exists():
            cleaned_csv_path = alt_path
            print(f"⚠️  Utilisation de {alt_path.name} à la place")
        else:
            raise FileNotFoundError(
                f"Fichier de données introuvable:\n"
                f"  Cherché: {cleaned_csv_path}\n"
                f"  Alternative: {alt_path}\n"
                f"  Contenu du dossier: {list(DATA_DIR.glob('*.csv'))}"
            )
    
    # Générer le chemin de sortie automatiquement
    if model_output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_path = MODELS_DIR / f"regressor_{model_version}_{timestamp}.pkl"  # ← Utilise MODELS_DIR
    else:
        model_output_path = Path(model_output_path)
    
    # Récupérer la configuration
    configs = get_model_configs()
    if model_version not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Version '{model_version}' inconnue. Disponibles: {available}")
    
    grid = configs[model_version]
    
    # Démarrer un run MLflow
    with mlflow.start_run(run_name=f"{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        print(f"\n{'='*70}")
        print(f"🎯 Entraînement de la version: {model_version.upper()}")
        print(f"{'='*70}")
        print(f"📁 Dossier projet: {PROJECT_ROOT}")
        print(f"📂 Données: {DATA_DIR}")
        print(f"💾 Modèles: {MODELS_DIR}")
        print(f"📊 MLflow: {MLRUNS_DIR}")
        print(f"{'='*70}\n")
        
        # Logger la version du modèle
        mlflow.log_param("model_version", model_version)
        mlflow.set_tag("model_version", model_version)
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("dataset", "car_prices")
        
        # 1. Charger et préparer données
        print(f"📂 Chargement des données: {cleaned_csv_path.name}")
        data = pd.read_csv(cleaned_csv_path)
        mlflow.log_param("dataset_size", len(data))
        mlflow.log_param("dataset_path", str(cleaned_csv_path))
        print(f"   ✓ {len(data)} lignes chargées")
        
        X, y, encoders = prepare_features_for_training(data.copy())
        
        # 2. Train/Test split
        test_size = 0.2
        random_state = 123
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        print(f"\n✂️ Split effectué:")
        print(f"   Train: {len(X_train)} lignes")
        print(f"   Test: {len(X_test)} lignes")
        
        # 3. Normalisation
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Logger les hyperparamètres
        print(f"\n⚙️ Hyperparamètres ({model_version}):")
        for param_name, param_value in grid.items():
            mlflow.log_param(f"grid_{param_name}", param_value[0])
            print(f"   {param_name}: {param_value[0]}")
        
        model = GridSearchCV(
            estimator=XGBRegressor(random_state=123, tree_method='hist'),
            param_grid=grid,
            scoring='neg_root_mean_squared_error',
            cv=5,
            verbose=0,
            n_jobs=-1
        )
        
        # 5. Fit
        print(f"\n🔄 Entraînement du modèle {model_version}...")
        model.fit(X_train_scaled, y_train)
        
        mlflow.log_params({f"best_{k}": v for k, v in model.best_params_.items()})
        
        # 6. Évaluation test
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print("\n📊 Métriques du modèle (test set):")
        print(f"   R² Score: {r2:.6f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        
        mlflow.log_metrics({
            "test_r2_score": r2,
            "test_mae": mae,
            "test_rmse": rmse
        })
        
        # 7. Réentraînement complet
        print("\n🔄 Réentraînement sur le dataset complet...")
        X_combined = np.vstack([X_train_scaled, X_test_scaled])
        y_combined = np.concatenate([y_train, y_test])
        model.fit(X_combined, y_combined)
        
        y_pred_combined = model.predict(X_combined)
        r2_combined = r2_score(y_combined, y_pred_combined)
        mae_combined = mean_absolute_error(y_combined, y_pred_combined)
        rmse_combined = np.sqrt(mean_squared_error(y_combined, y_pred_combined))
        
        print("\n📊 Métriques finales (full dataset):")
        print(f"   R² Score: {r2_combined:.6f}")
        print(f"   MAE: {mae_combined:.2f}")
        print(f"   RMSE: {rmse_combined:.2f}")
        
        mlflow.log_metrics({
            "final_r2_score": r2_combined,
            "final_mae": mae_combined,
            "final_rmse": rmse_combined
        })
        
        # 8. Sauvegarde
        model_data = {
            'model': model,
            'scaler': scaler,
            'encoders': encoders,
            'version': model_version,
            'timestamp': datetime.now().isoformat(),
            'training_data_path': str(cleaned_csv_path)
        }
        
        with open(model_output_path, 'wb') as file:
            pickle.dump(model_data, file)
        
        mlflow.xgboost.log_model(model.best_estimator_, "xgboost_model")
        mlflow.log_artifact(str(model_output_path), "model_pickle")
        
        print(f"\n✅ Modèle sauvegardé: {model_output_path}")
        print(f"📊 MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"🔗 Voir les résultats: mlflow ui")
        print(f"\n{'='*70}\n")
        
        return model_data, model_output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        version = sys.argv[1]
    else:
        version = 'baseline'
    
    print(f"\n{'='*70}")
    print(f"🚀 MLOPS PIPELINE - ENTRAÎNEMENT DE MODÈLE")
    print(f"{'='*70}")
    print(f"📁 Environment: {'Docker' if os.path.exists('/app') else 'Local'}")
    print(f"📁 Dossier projet: {PROJECT_ROOT}")
    print(f"📂 Données: {DATA_DIR}")
    print(f"💾 Modèles: {MODELS_DIR}")
    print(f"\nVersion demandée: {version}")
    print(f"Versions disponibles: {', '.join(get_model_configs().keys())}")
    
    # Debug: afficher les fichiers disponibles
    csv_files = list(DATA_DIR.glob('*.csv'))
    print(f"\nFichiers CSV trouvés: {[f.name for f in csv_files]}")
    print(f"{'='*70}\n")
    
    try:
        model_data, model_path = train_model(model_version=version)
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCÈS - Modèle {version} entraîné!")
        print(f"{'='*70}")
        print(f"📦 Fichier: {model_path}")
        print(f"💡 Pour tester: python predict.py")
        print(f"📊 Pour voir MLflow: mlflow ui")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}\n")
        import traceback
        traceback.print_exc()
        raise
