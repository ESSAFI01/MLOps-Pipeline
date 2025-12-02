import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import logging
from datetime import datetime
import json

<<<<<<< HEAD
def load_model(model_path=r'C:\Users\Ayoub Gorry\Desktop\mlops\MLOps-Pipeline\Mlpro\models\regressorfinal.pkl'):
    """
    Charge le modèle, scaler et encoders
    """
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
    
    print("✅ Modèle chargé avec succès")
    return model_data
=======
# Détection automatique de l'environnement (comme train.py)
if os.path.exists('/app'):  # Dans Docker
    PROJECT_ROOT = Path('/app')
else:  # En local
    PROJECT_ROOT = Path(__file__).parent.parent
>>>>>>> 74abcf42d8a695cef55fa89bb4918e5372a6ce36

MODELS_DIR = PROJECT_ROOT / "Mlpro" / "models"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_latest_model(version=None):
    """Trouve le dernier modèle pour une version donnée"""
    if version:
        pattern = f"regressor_{version}_*.pkl"
    else:
        pattern = "regressor_*.pkl"
    
    models = list(MODELS_DIR.glob(pattern))
    
    if not models:
        raise FileNotFoundError(
            f"❌ Aucun modèle trouvé dans {MODELS_DIR}\n"
            f"   Pattern recherché: {pattern}\n"
            f"   Contenu du dossier: {list(MODELS_DIR.glob('*.pkl'))}\n"
            f"   💡 Lance d'abord: python train.py baseline"
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
        
        logger.info(f"✅ Modèle chargé depuis: {model_path.name}")
        if 'version' in model_data:
            logger.info(f"   Version: {model_data['version']}")
        if 'timestamp' in model_data:
            logger.info(f"   Timestamp: {model_data['timestamp']}")
        
        return model_data
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
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
            logger.warning(json.dumps(log_data))
        
        return None, error_msg
    
    # Normaliser et prédire
    normalized_data = scaler.transform(new_data)
    price = model.predict(normalized_data)[0]
    
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
        logger.info(json.dumps(log_data))
    
    return price, None


def predict_multiple(test_configs, model_data):
    """Prédictions multiples avec statistiques"""
    logger.info(f"🔄 Début prédictions batch: {len(test_configs)} voitures")
    
    results = []
    success_count = 0
    error_count = 0
    
    for config in test_configs:
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
    
    logger.info(f"✅ Batch terminé: {success_count} succès, {error_count} erreurs")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🚀 Démarrage du système de prédiction")
    logger.info(f"📁 Environment: {'Docker' if os.path.exists('/app') else 'Local'}")
    logger.info(f"📁 Dossier projet: {PROJECT_ROOT}")
    logger.info(f"💾 Dossier modèles: {MODELS_DIR}")
    logger.info("=" * 50)
    
    try:
        model_data = load_model()
        
        # Test avec quelques voitures
        test_configs = [
            {'name': 'Ford Fiesta', 'manufacturer': 'FORD', 'age': 5, 
             'kilometerage': 50000.0, 'engine': 'Petrol', 'transmission': 'Automatic'},
            {'name': 'Vauxhall Corsa', 'manufacturer': 'VAUXHALL', 'age': 3, 
             'kilometerage': 30000.0, 'engine': 'Petrol', 'transmission': 'Manual'},
            {'name': 'Bmw 3 Series', 'manufacturer': 'BMW', 'age': 2, 
             'kilometerage': 20000.0, 'engine': 'Diesel', 'transmission': 'Automatic'},
        ]
        
        print("\n🚗 Résultats des prédictions:\n")
        results = predict_multiple(test_configs, model_data)
        print(results.to_string(index=False))
        print("\n" + "=" * 50 + "\n")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"\n{e}")
        print("\n💡 Solution: Lance d'abord l'entraînement:")
        print("   python train.py baseline\n")
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
