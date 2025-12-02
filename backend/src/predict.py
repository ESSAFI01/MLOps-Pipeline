import pandas as pd
import numpy as np
import pickle

def load_model(model_path=r'C:\Users\Ayoub Gorry\Desktop\mlops\MLOps-Pipeline\Mlpro\models\regressorfinal.pkl'):
    """
    Charge le modèle, scaler et encoders
    """
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
    
    print("✅ Modèle chargé avec succès")
    return model_data

def predict_price(car_config, model_data):
    """
    Prédit le prix d'une voiture
    
    car_config: dict avec {
        'name': str,
        'manufacturer': str,
        'age': int,
        'kilometerage': float,
        'engine': str,
        'transmission': str
    }
    """
    model = model_data['model']
    scaler = model_data['scaler']
    encoders = model_data['encoders']
    
    # Encoder les features
    try:
        new_data = np.zeros((1, 6))
        new_data[0, 0] = encoders['le_name'].transform([car_config['name']])[0]
        new_data[0, 1] = encoders['le_manufacturer'].transform([car_config['manufacturer']])[0]
        new_data[0, 2] = float(car_config['age'])
        new_data[0, 3] = float(car_config['kilometerage'])
        new_data[0, 4] = encoders['le_engine'].transform([car_config['engine']])[0]
        new_data[0, 5] = encoders['le_transmission'].transform([car_config['transmission']])[0]
    except ValueError as e:
        return None, f"⚠️ Catégorie inconnue : {e}"
    
    # Normaliser
    normalized_data = scaler.transform(new_data)
    
    # Prédire
    price = model.predict(normalized_data)[0]
    
    return price, None

def predict_multiple(test_configs, model_data):
    """
    Prédictions multiples
    """
    results = []
    for config in test_configs:
        price, error = predict_price(config, model_data)
        if error:
            print(error)
            continue
        
        results.append({
            'Car': config['name'],
            'Manufacturer': config['manufacturer'],
            'Age': config['age'],
            'Mileage': f"{config['kilometerage']:.0f}",
            'Engine': config['engine'],
            'Transmission': config['transmission'],
            'Estimated Price': f"{price:.0f} EUR"
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Exemple d'utilisation
    model_data = load_model()
    
    test_configs = [
        {'name': 'Ford Fiesta', 'manufacturer': 'FORD', 'age': 5, 
         'kilometerage': 5000.0, 'engine': 'Petrol', 'transmission': 'Automatic'},
        {'name': 'Vauxhall Corsa', 'manufacturer': 'VAUXHALL', 'age': 3, 
         'kilometerage': 30000.0, 'engine': 'Petrol', 'transmission': 'Manual'},
    ]
    
    
    results = predict_multiple(test_configs, model_data)
    print("\n🚗 Prédictions :")
    print(results)
