import mlflow
import mlflow.sklearn
import pickle

with mlflow.start_run():

    # ton code d'entraînement ici...

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("model.pkl")
