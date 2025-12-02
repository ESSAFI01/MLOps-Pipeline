"""
Script pour monitorer les prédictions avec MLflow
"""
import mlflow
from datetime import datetime, timedelta
import pandas as pd

def monitor_predictions():
    """Monitorer les prédictions récentes dans MLflow"""
    # Configurer l'URI de MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Récupérer l'expérience
    experiment = mlflow.get_experiment_by_name("CarPricePredictions")
    if experiment is None:
        print("⚠️ Aucune expérience 'CarPricePredictions' trouvée")
        return
    
    # Récupérer les dernières runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=10
    )
    
    print(f"🔍 Dernières prédictions ({len(runs)} runs trouvées):")
    print("=" * 80)
    
    for idx, run in runs.iterrows():
        print(f"\n Run: {run['run_id']}")
        print(f"Date: {run['start_time']}")
        print(f"Taille batch: {run['params.batch_size'] if 'params.batch_size' in run else 'N/A'}")
        print(f"Succès: {run['metrics.success_count'] if 'metrics.success_count' in run else 'N/A'}")
        print(f"Erreurs: {run['metrics.error_count'] if 'metrics.error_count' in run else 'N/A'}")
        print(f"Taux succès: {float(run['metrics.success_rate']) * 100:.1f}%" if 'metrics.success_rate' in run else '   📈 Taux succès: N/A')
        print(f"Prix moyen: {run['metrics.avg_predicted_price']:.0f} €" if 'metrics.avg_predicted_price' in run else '   💰 Prix moyen: N/A')
        print(f"MLflow UI: http://localhost:5000/#/experiments/{experiment.experiment_id}/runs/{run['run_id']}")
    
    print("\n" + "=" * 80)
    

if __name__ == "__main__":
    monitor_predictions()