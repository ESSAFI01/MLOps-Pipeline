import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from data_utils import prepare_features_for_training, scale_features

def train_model(cleaned_csv_path=r'Mlpro\dataSet\cleaned_cardata3.csv', 
                model_output_path=r'Mlpro\models\regressorfinal.pkl'):
    """
    Entraîne le modèle XGBoost et sauvegarde le pipeline complet
    """
    # 1. Charger et préparer données
    data = pd.read_csv(cleaned_csv_path)
    X, y, encoders = prepare_features_for_training(data.copy())
    
    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    
    # 3. Normalisation
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 4. GridSearch XGBoost (hyperparamètres optimisés)
    grid = {
        'colsample_bytree': [0.7],
        'learning_rate': [0.1],
        'max_depth': [9],
        'min_child_weight': [1],
        'n_estimators': [350],
        'subsample': [0.8],
        'gamma': [0],
        'reg_alpha': [0.1],
        'reg_lambda': [1.5]
    }
    
    model = GridSearchCV(
        estimator=XGBRegressor(random_state=123, tree_method='hist'),
        param_grid=grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    # 5. Fit sur données train
    print("🔄 Entraînement du modèle...")
    model.fit(X_train_scaled, y_train, verbose=1)
    
    # 6. Évaluation
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n📊 Métriques du modèle (test set):")
    print(f"  R² Score: {r2:.6f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    
    # 7. Réentraînement sur tout le dataset (optionnel mais tu le fais)
    print("\n🔄 Réentraînement sur le dataset complet...")
    X_combined = np.vstack([X_train_scaled, X_test_scaled])
    y_combined = np.concatenate([y_train, y_test])
    model.fit(X_combined, y_combined, verbose=1)
    
    y_pred_combined = model.predict(X_combined)
    r2_combined = r2_score(y_combined, y_pred_combined)
    mae_combined = mean_absolute_error(y_combined, y_pred_combined)
    rmse_combined = np.sqrt(mean_squared_error(y_combined, y_pred_combined))
    
    print("\n📊 Métriques finales (full dataset):")
    print(f"  R² Score: {r2_combined:.6f}")
    print(f"  MAE: {mae_combined:.2f}")
    print(f"  RMSE: {rmse_combined:.2f}")
    
    # 8. Sauvegarde du pipeline complet
    model_data = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders
    }
    
    with open(model_output_path, 'wb') as file:
        pickle.dump(model_data, file)
    
    print(f"\n✅ Modèle sauvegardé : {model_output_path}")
    return model_data

if __name__ == "__main__":
    train_model()
