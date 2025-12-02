"""
Configuration centralisée MLflow pour le projet Car Price Prediction
"""
import os
from pathlib import Path

# Détection automatique de l'environnement
if os.path.exists('/app'):  # Dans Docker
    PROJECT_ROOT = Path('/app')
else:  # En local
    CURRENT_DIR = Path(__file__).parent
    PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == 'src' else CURRENT_DIR

# Chemins
MLRUNS_DIR = PROJECT_ROOT.parent / "mlruns"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "dataSet"
PREDICTIONS_LOG_DIR = PROJECT_ROOT / "predictionsLog"
PREDICTIONS_CSV_DIR = PROJECT_ROOT / "predictionsCSV"
PREDICTIONS_HTML_DIR = PROJECT_ROOT / "predictionsHtml"

# Créer les dossiers s'ils n'existent pas
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_CSV_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_HTML_DIR.mkdir(parents=True, exist_ok=True)

# Configuration MLflow
MLFLOW_CONFIG = {
    # Nom de l'expérience UNIFIÉ
    "EXPERIMENT_NAME": "car_price_prediction",
    
    # URI de tracking
    "TRACKING_URI_FILE": f"file:{MLRUNS_DIR.as_posix()}",
    "TRACKING_URI_HTTP": "http://127.0.0.1:5000",  # Pour l'UI
    
    # Noms des modèles
    "REGISTERED_MODEL_NAME": "car_price_model",
    
    # Tags par défaut
    "DEFAULT_TAGS": {
        "project": "car_price_prediction",
        "framework": "scikit-learn",
        "model_type": "XGBoost",
        "task": "regression",
        "dataset": "car_prices"
    },
    
    # Chemins
    "PATHS": {
        "mlruns": MLRUNS_DIR,
        "models": MODELS_DIR,
        "data": DATA_DIR,
        "project_root": PROJECT_ROOT,
        "predictions_log": PREDICTIONS_LOG_DIR,
        "predictions_csv": PREDICTIONS_CSV_DIR,
        "predictions_html": PREDICTIONS_HTML_DIR
    }
}


def setup_mlflow(mode='train'):
    """
    Configure MLflow pour l'entraînement ou la prédiction
    
    Args:
        mode (str): 'train' ou 'predict'
    
    Returns:
        mlflow module configuré
    """
    import mlflow
    
    if mode == 'train':
        # Mode entraînement - stockage local
        mlflow.set_tracking_uri(MLFLOW_CONFIG["TRACKING_URI_FILE"])
    else:
        # Mode prédiction - connexion au serveur
        mlflow.set_tracking_uri(MLFLOW_CONFIG["TRACKING_URI_HTTP"])
    
    # Définir l'expérience (créée si elle n'existe pas)
    mlflow.set_experiment(MLFLOW_CONFIG["EXPERIMENT_NAME"])
    
    return mlflow


def get_project_paths():
    """Retourne tous les chemins du projet"""
    return MLFLOW_CONFIG["PATHS"]