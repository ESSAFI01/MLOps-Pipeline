import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_clean_data(raw_csv_path):
    """
    Charge et nettoie les données brutes
    """
    data = pd.read_csv(raw_csv_path)
    
    # Supprimer les NaN transmission
    data = data.dropna()
    data = data.drop_duplicates(keep='first', inplace=False)
    
    # Nettoyer les colonnes
    data.columns = data.columns.str.strip()
    
    return data

def extract_manufacturer(data):
    """
    Extrait le manufacturer depuis le nom du véhicule
    """
    data['manufacturer'] = data['name'].apply(
        lambda x: 'LAND ROVER' if 'LAND ROVER' in x.upper() else x.split()[0].upper()
    )
    data['name'] = data['name'].str.title()
    return data

def clean_price(data):
    """
    Nettoie et convertit la colonne price
    """
    data['price'] = data['price'].str.replace(',', '').str.replace('£', '').str.extract('(\d+)').astype(np.int64)
    return data

def create_age_feature(data, current_year=2025):
    """
    Crée la feature 'age' à partir de year
    """
    data['year'] = data['year'].astype(str).astype(np.int64)
    data['age'] = current_year - data['year']
    data['age'] = data['age'].astype(np.int64)
    
    # Supprimer age < 0
    data = data[data['age'] >= 0]
    return data

def clean_kilometerage(data):
    """
    Convertit mileage en kilometerage
    """
    data['mileage'] = data['mileage'].str.replace(',', '').str.replace(' ', '').str.replace('miles', '').astype(np.int64)
    data['mileage'] = (data['mileage'] * 1.60934).astype(np.int64)
    data.rename(columns={'mileage': 'kilometerage'}, inplace=True)
    return data

def clean_engine(data):
    """
    Normalise les valeurs engine
    """
    data['engine'] = data['engine'].str.strip()
    
    replacements = {
        'Petrol hybrid': 'Hybrid',
        'Petrol electric hy': 'Hybrid',
        'Petrol plug-in hybri': 'Pluginhybrid',
        'Hybrid electric': 'Hybrid',
        'Petrol/electric hybr': 'Hybrid',
        'Petrol/mhev': 'Hybrid',
        'Plug-in hybrid': 'Pluginhybrid',
        'Electric/diesel': 'Hybrid',
        'Electric only': 'Electric',
        'Petrol series phev': 'Hybrid',
        'Diesel hybrid': 'Hybrid',
        'Petrol parallel phev': 'Hybrid',
        'Petrol/plugin elec h': 'Hybrid',
        'Diesel/mhev': 'Hybrid',
        'Petrol/plugin e': 'Pluginhybrid',
        'Diesel/electric hybr': 'Hybrid',
        'Diesel plug-in hybri': 'Pluginhybrid',
        'Petrol plug-in': 'Pluginhybrid',
        'Diesel/plugin e': 'Pluginhybrid',
        'Diesel parallel phev': 'Hybrid',
        'Diesel electric hy': 'Hybrid',
        'Diesel/plugin elec h': 'Pluginhybrid',
        'Electric/petrol': 'Hybrid',
        'Petrol/electric': 'Hybrid'
    }
    data = data.replace({'engine': replacements})
    
    # Supprimer lignes avec Unleaded, Na, Bi fuel
    data = data[~data['engine'].isin(['Unleaded', 'Na', 'Bi fuel'])]
    return data

def clean_transmission(data):
    """
    Normalise transmission
    """
    replacements = {
        'Semi auto': 'Semiautomatic',
        'Semi-auto': 'Semiautomatic',
        'Semi automatic': 'Semiautomatic',
        'Semi-automatic': 'Semiautomatic',
        'Cvt automa': 'Automatic',
        'Cvt': 'Automatic',
        'Tr-ew': 'Automatic',
        'Tr-ga': 'Automatic',
        'Tr-a7': 'Automatic',
        'Tr-ai': 'Automatic',
        'Auto (10 gears)': 'Automatic',
        'Tr-wa': 'Automatic',
        'Auto (7 gears)': 'Automatic'
    }
    data = data.replace({'transmission': replacements})
    
    # Supprimer Unknown et Other
    data = data[~data['transmission'].isin(['Unknown', 'Other'])]
    return data

def filter_sufficient_models(data, min_count=10):
    """
    Garde uniquement les noms de voitures avec au moins min_count occurrences
    """
    name_counts = data['name'].value_counts()
    data = data[data['name'].isin(name_counts[name_counts >= min_count].index)]
    return data

def reorder_columns(data):
    """
    Réordonne les colonnes pour clarté
    """
    return data[['name', 'manufacturer', 'year', 'age', 'kilometerage', 'engine', 'transmission', 'price']]

def full_clean_pipeline(raw_csv_path, output_csv_path):
    """
    Pipeline complet de nettoyage
    """
    data = load_and_clean_data(raw_csv_path)
    data = extract_manufacturer(data)
    data = clean_price(data)
    data = create_age_feature(data)
    data = clean_kilometerage(data)
    data = clean_engine(data)
    data = clean_transmission(data)
    data = filter_sufficient_models(data)
    data = reorder_columns(data)
    
    # Sauvegarder
    data.to_csv(output_csv_path, index=False)
    print(f"✅ Données nettoyées sauvegardées : {output_csv_path}")
    return data

def prepare_features_for_training(data):
    """
    Encode les features catégorielles et crée X, y
    """
    # Label encoding
    le_name = LabelEncoder()
    le_manufacturer = LabelEncoder()
    le_engine = LabelEncoder()
    le_transmission = LabelEncoder()
    
    data['name'] = le_name.fit_transform(data['name'])
    data['manufacturer'] = le_manufacturer.fit_transform(data['manufacturer'])
    data['engine'] = le_engine.fit_transform(data['engine'])
    data['transmission'] = le_transmission.fit_transform(data['transmission'])
    
    # Supprimer 'year' (on garde 'age')
    data = data.drop('year', axis=1)
    
    # X, y
    X = data.drop('price', axis=1)
    y = data['price'] / 10  # Division par 10 comme dans ton code
    
    return X, y, {
        'le_name': le_name,
        'le_manufacturer': le_manufacturer,
        'le_engine': le_engine,
        'le_transmission': le_transmission
    }

def scale_features(X_train, X_test):
    """
    Normalise les features avec MinMaxScaler
    """
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
