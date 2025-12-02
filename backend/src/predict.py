import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import logging
from datetime import datetime
import json
import mlflow
import mlflow.sklearn

# Détection automatique de l'environnement (comme train.py)
if os.path.exists('/app'):  # Dans Docker
    PROJECT_ROOT = Path('/app')
else:  # En local
    PROJECT_ROOT = Path(__file__).parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # URI de votre serveur MLflow

# Configuration du logging avec encodage UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'predictionsLog/predictions.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour gérer les types numpy"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def find_latest_model(version=None):
    """Trouve le dernier modèle pour une version donnée"""
    if version:
        pattern = f"regressor_{version}_*.pkl"
    else:
        pattern = "regressor_*.pkl"
    
    models = list(MODELS_DIR.glob(pattern))
    
    if not models:
        raise FileNotFoundError(
            f"Aucun modèle trouvé dans {MODELS_DIR}\n"
            f"Pattern recherché: {pattern}\n"
            f"Contenu du dossier: {list(MODELS_DIR.glob('*.pkl'))}\n"
            f"Lance d'abord: python train.py baseline"
        )
    
    # Trier par date de modification (le plus récent en premier)
    latest = max(models, key=lambda p: p.stat().st_mtime)
    return latest


def load_model(model_path=None):
    """Charge le modèle, scaler et encoders"""
    try:
        if model_path is None:
            model_path = find_latest_model()
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")
        
        with open(model_path, 'rb') as file:
            model_data = pickle.load(file)
        
        logger.info(f"Modèle chargé depuis: {model_path.name}")
        if 'version' in model_data:
            logger.info(f"Version: {model_data['version']}")
        if 'timestamp' in model_data:
            logger.info(f"Timestamp: {model_data['timestamp']}")
        
        return model_data
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}")
        raise


def predict_price(car_config, model_data, log_prediction=True):
    """Prédit le prix d'une voiture avec logging"""
    start_time = datetime.now()
    
    model = model_data['model']
    scaler = model_data['scaler']
    encoders = model_data['encoders']
    
    try:
        new_data = np.zeros((1, 6))
        new_data[0, 0] = encoders['le_name'].transform([car_config['name']])[0]
        new_data[0, 1] = encoders['le_manufacturer'].transform([car_config['manufacturer']])[0]
        new_data[0, 2] = float(car_config['age'])
        new_data[0, 3] = float(car_config['kilometerage'])
        new_data[0, 4] = encoders['le_engine'].transform([car_config['engine']])[0]
        new_data[0, 5] = encoders['le_transmission'].transform([car_config['transmission']])[0]
    except ValueError as e:
        error_msg = f"Catégorie inconnue: {e}"
        
        if log_prediction:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'input': car_config,
                'status': 'error',
                'error': error_msg
            }
            logger.warning(json.dumps(log_data, cls=CustomJSONEncoder))
        
        return None, error_msg
    
    # Normaliser et prédire
    normalized_data = scaler.transform(new_data)
    price = float(model.predict(normalized_data)[0])
    
    # Calculer le temps de prédiction
    prediction_time = (datetime.now() - start_time).total_seconds()
    
    # Logger la prédiction réussie
    if log_prediction:
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'input': car_config,
            'predicted_price_EUR': round(price, 2),
            'predicted_price_MAD': round(price * 10, 2),
            'prediction_time_ms': round(prediction_time * 1000, 2),
            'status': 'success'
        }
        logger.info(json.dumps(log_data, cls=CustomJSONEncoder))
    
    return price, None


def predict_multiple(test_configs, model_data, mlflow_experiment_name="predictions"):
    """Prédictions multiples avec tracking MLflow"""
    logger.info(f"Debut predictions batch: {len(test_configs)} voitures")
    
    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(mlflow_experiment_name)
    
    results = []
    success_count = 0
    error_count = 0
    all_predictions = []
    
    # Démarrer une run MLflow
    with mlflow.start_run(run_name=f"batch_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Loguer les métadonnées du modèle
        if 'version' in model_data:
            mlflow.log_param("model_version", model_data['version'])
        if 'timestamp' in model_data:
            mlflow.log_param("model_timestamp", model_data['timestamp'])
        
        mlflow.log_param("batch_size", len(test_configs))
        mlflow.log_param("run_id", run.info.run_id)
        
        for i, config in enumerate(test_configs):
            price, error = predict_price(config, model_data, log_prediction=True)
            
            if error:
                error_count += 1
                results.append({
                    'Car': config['name'],
                    'Status': 'Error',
                    'Message': error
                })
            else:
                success_count += 1
                results.append({
                    'Car': config['name'],
                    'Manufacturer': config['manufacturer'],
                    'Age': config['age'],
                    'Mileage': f"{config['kilometerage']:.0f}",
                    'Engine': config['engine'],
                    'Transmission': config['transmission'],
                    'Estimated Price (EUR)': f"{price:.0f}",
                    'Estimated Price (MAD)': f"{price * 10:.0f}"
                })
                
                # Stocker pour les métriques
                all_predictions.append({
                    'config': config,
                    'predicted_price': price
                })
        
        # Loguer les métriques globales
        mlflow.log_metric("success_count", success_count)
        mlflow.log_metric("error_count", error_count)
        mlflow.log_metric("success_rate", success_count / len(test_configs))
        
        if success_count > 0:
            prices = [p['predicted_price'] for p in all_predictions]
            mlflow.log_metric("avg_predicted_price", np.mean(prices))
            mlflow.log_metric("min_predicted_price", np.min(prices))
            mlflow.log_metric("max_predicted_price", np.max(prices))
            mlflow.log_metric("std_predicted_price", np.std(prices))
        
        # Sauvegarder les résultats dans un fichier et le loguer dans MLflow
        results_df = pd.DataFrame(results)
        results_file = PROJECT_ROOT / f"predictionsCSV/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Loguer le fichier de résultats
        mlflow.log_artifact(str(results_file), "predictions")
        
        # Loguer un échantillon des prédictions comme table
        sample_predictions = []
        for pred in all_predictions[:5]:  # Les 5 premières prédictions
            sample_predictions.append({
                "name": pred['config']['name'],
                "manufacturer": pred['config']['manufacturer'],
                "age": pred['config']['age'],
                "kilometerage": pred['config']['kilometerage'],
                "engine": pred['config']['engine'],
                "transmission": pred['config']['transmission'],
                "predicted_price_eur": round(pred['predicted_price'], 2),
                "predicted_price_mad": round(pred['predicted_price'] * 10, 2)
            })
        
        # Créer et loguer une table HTML des prédictions
        if sample_predictions:
            html_table = pd.DataFrame(sample_predictions).to_html(index=False)
            html_file = PROJECT_ROOT / "predictionsHtml/predictions_sample.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"<h3>Prédictions d'échantillon - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>")
                f.write(html_table)
            mlflow.log_artifact(str(html_file), "predictions")
        
        logger.info(f"Batch termine: {success_count} succes, {error_count} erreurs")
        logger.info(f"Resultats sauvegardes dans: {results_file}")
        logger.info(f"MLflow Run: {run.info.run_id}")
        
        # Afficher l'URL MLflow
        mlflow_ui_url = f"{MLFLOW_TRACKING_URI}/#/experiments/{mlflow.get_experiment_by_name(mlflow_experiment_name).experiment_id}/runs/{run.info.run_id}"
        logger.info(f"Voir les resultats dans MLflow UI: {mlflow_ui_url}")
    
    return results_df


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Demarrage du systeme de prediction avec MLflow")
    logger.info(f"Environment: {'Docker' if os.path.exists('/app') else 'Local'}")
    logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Dossier projet: {PROJECT_ROOT}")
    logger.info(f"Dossier modeles: {MODELS_DIR}")
    logger.info("=" * 50)
    
    try:
        model_data = load_model()
        
        # Test avec quelques voitures
        test_configs = [
            {'name': 'Ford Fiesta', 'manufacturer': 'FORD', 'age': 5, 
             'kilometerage': 5000.0, 'engine': 'Petrol', 'transmission': 'Automatic'},
            {'name': 'Vauxhall Corsa', 'manufacturer': 'VAUXHALL', 'age': 3, 
             'kilometerage': 30.0, 'engine': 'Petrol', 'transmission': 'Manual'},
            {'name': 'Bmw 3 Series', 'manufacturer': 'BMW', 'age': 2, 
             'kilometerage': 200.0, 'engine': 'Diesel', 'transmission': 'Automatic'},
            {'name': 'Audi A4', 'manufacturer': 'AUDI', 'age': 4, 
             'kilometerage': 400000.0, 'engine': 'Diesel', 'transmission': 'Automatic'},
            {'name': 'Mercedes C Class', 'manufacturer': 'MERCEDES', 'age': 1, 
             'kilometerage': 10330.0, 'engine': 'Petrol', 'transmission': 'Manual'},
        ]
        
        print("\nResultats des predictions:\n")
        results = predict_multiple(test_configs, model_data, mlflow_experiment_name="CarPricePredictions")
        print(results.to_string(index=False))
        print("\n" + "=" * 50 + "\n")
        
        # Afficher les liens MLflow
        print("📊 MLflow Tracking:")
        print(f"   Interface: {MLFLOW_TRACKING_URI}")
        print("   Pour demarrer l'interface MLflow: mlflow ui --host 0.0.0.0 --port 5000")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n{e}")
        print("\nSolution: Lance d'abord l'entrainement:")
        print("   python train.py baseline\n")
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()