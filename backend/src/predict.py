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
from mlflow_config import MLFLOW_CONFIG, setup_mlflow, get_project_paths

# Utiliser les chemins depuis la config centralisée
PATHS = get_project_paths()
PROJECT_ROOT = PATHS["project_root"]
MODELS_DIR = PATHS["models"]
PREDICTIONS_LOG_DIR = PATHS["predictions_log"]
PREDICTIONS_CSV_DIR = PATHS["predictions_csv"]
PREDICTIONS_HTML_DIR = PATHS["predictions_html"]

# Configuration du logging avec encodage UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PREDICTIONS_LOG_DIR / 'predictions.log', encoding='utf-8'),
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
        if 'mlflow_experiment' in model_data:
            logger.info(f"Expérience MLflow: {model_data['mlflow_experiment']}")
        
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


def predict_multiple(test_configs, model_data, mlflow_experiment_name=None):
    """Prédictions multiples avec tracking MLflow"""
    logger.info(f"Début prédictions batch: {len(test_configs)} voitures")
    
    # Configuration MLflow UNIFIÉE
    mlflow = setup_mlflow(mode='predict')
    
    # Utiliser le nom d'expérience centralisé si non spécifié
    if mlflow_experiment_name is None:
        mlflow_experiment_name = MLFLOW_CONFIG["EXPERIMENT_NAME"]
    else:
        mlflow_experiment_name = mlflow_experiment_name.strip()
    
    # S'assurer qu'on utilise la bonne expérience
    if mlflow_experiment_name != MLFLOW_CONFIG["EXPERIMENT_NAME"]:
        logger.warning(f"Expérience spécifiée '{mlflow_experiment_name}' différente de l'expérience centralisée '{MLFLOW_CONFIG['EXPERIMENT_NAME']}'")
    
    mlflow.set_experiment(mlflow_experiment_name)
    
    results = []
    success_count = 0
    error_count = 0
    all_predictions = []
    
    # Démarrer une run MLflow
    with mlflow.start_run(run_name=f"batch_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info(f"Expérience MLflow: '{mlflow_experiment_name}'")
        
        # Logger les tags par défaut
        default_tags = MLFLOW_CONFIG["DEFAULT_TAGS"]
        for tag_name, tag_value in default_tags.items():
            mlflow.set_tag(tag_name, tag_value)
        
        # Logger les métadonnées du modèle
        if 'version' in model_data:
            mlflow.log_param("model_version", model_data['version'])
            mlflow.set_tag("model_version", model_data['version'])
        if 'timestamp' in model_data:
            mlflow.log_param("model_timestamp", model_data['timestamp'])
        if 'mlflow_experiment' in model_data:
            mlflow.log_param("training_experiment", model_data['mlflow_experiment'])
        
        mlflow.log_param("batch_size", len(test_configs))
        mlflow.log_param("run_id", run.info.run_id)
        mlflow.log_param("current_experiment", mlflow_experiment_name)
        
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = PREDICTIONS_CSV_DIR / f"predictions_{timestamp}.csv"
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False, encoding='utf-8')
        
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
            html_file = PREDICTIONS_HTML_DIR / f"predictions_sample_{timestamp}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"<h3>Prédictions d'échantillon - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>")
                f.write(f"<p>Expérience: <strong>{mlflow_experiment_name}</strong></p>")
                f.write(f"<p>Modèle: <strong>{model_data.get('version', 'N/A')}</strong></p>")
                f.write(html_table)
            mlflow.log_artifact(str(html_file), "predictions")
        
        logger.info(f"Batch terminé: {success_count} succès, {error_count} erreurs")
        logger.info(f"Résultats sauvegardés dans: {results_file}")
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Afficher l'URL MLflow
        try:
            experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
            if experiment:
                mlflow_ui_url = f"{MLFLOW_CONFIG['TRACKING_URI_HTTP']}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
                logger.info(f"Voir les résultats dans MLflow UI: {mlflow_ui_url}")
        except:
            logger.info(f"Interface MLflow: {MLFLOW_CONFIG['TRACKING_URI_HTTP']}")
    
    return results_df


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Démarrage du système de prédiction avec MLflow")
    logger.info(f"Environnement: {'Docker' if os.path.exists('/app') else 'Local'}")
    logger.info(f"MLflow Tracking URI: {MLFLOW_CONFIG['TRACKING_URI_HTTP']}")
    logger.info(f"Dossier projet: {PROJECT_ROOT}")
    logger.info(f"Dossier modèles: {MODELS_DIR}")
    logger.info(f"Expérience MLflow: '{MLFLOW_CONFIG['EXPERIMENT_NAME']}'")
    logger.info("=" * 60)
    
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
        
        print(f"\n🎯 Prédictions avec l'expérience: '{MLFLOW_CONFIG['EXPERIMENT_NAME']}'")
        print(f"📊 Modèle chargé: {model_data.get('version', 'N/A')}")
        print("=" * 60 + "\n")
        
        results = predict_multiple(test_configs, model_data, mlflow_experiment_name=MLFLOW_CONFIG["EXPERIMENT_NAME"])
        print(results.to_string(index=False))
        print("\n" + "=" * 60 + "\n")
        
        # Afficher les liens MLflow
        print("📊 MLflow Tracking:")
        print(f"   Interface: {MLFLOW_CONFIG['TRACKING_URI_HTTP']}")
        print(f"   Expérience: '{MLFLOW_CONFIG['EXPERIMENT_NAME']}'")
        print("\n💡 Pour démarrer l'interface MLflow:")
        print("   mlflow ui --host 0.0.0.0 --port 5000")
        print("\n📁 Dossiers créés:")
        print(f"   Logs: {PREDICTIONS_LOG_DIR}")
        print(f"   CSV: {PREDICTIONS_CSV_DIR}")
        print(f"   HTML: {PREDICTIONS_HTML_DIR}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n❌ {e}")
        print("\n💡 Solution: Lancez d'abord l'entraînement:")
        print("   python train.py baseline\n")
        print("Ou pour une version spécifique:")
        print("   python train.py fast_model")
        print("   python train.py accurate_model\n")
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ Erreur: {e}")