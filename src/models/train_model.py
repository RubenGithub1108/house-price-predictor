import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
import yaml
import logging
import platform
import sklearn
import os
import warnings

# Ignorar advertencias de obsolescencia para una salida más limpia
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------
# Configuración
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Argumentos
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Entrena y evalúa el mejor modelo usando los datos de train/test.")
    parser.add_argument("--config-path", type=str, required=True, help="Ruta al best_model_config.yaml")
    parser.add_argument("--data-dir", type=str, required=True, help="Directorio con los datos procesados (X_train.csv, etc.)")
    parser.add_argument("--models-dir", type=str, required=True, help="Directorio para guardar el modelo final entrenado")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="URI de tracking de MLflow")
    return parser.parse_args()

# -----------------------------
# Funciones auxiliares
# -----------------------------
def get_model_instance(name, params):
    clean_params = {k: v for k, v in params.items() if not isinstance(v, (dict, list))}
    model_map = {
        'LinearRegression': LinearRegression, 'Ridge': Ridge,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'XGBoost': xgb.XGBRegressor
    }
    if name not in model_map:
        raise ValueError(f"Modelo no soportado: {name}")
    model = model_map[name](); model.set_params(**clean_params)
    return model

# -----------------------------
# Lógica principal
# -----------------------------
def main(args):
    # Cargar configuración del mejor modelo
    logger.info(f"Cargando configuración desde: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Cargar los datos de entrenamiento y prueba ya divididos y procesados
    # ***** LÍNEA CORREGIDA *****
    logger.info(f"Cargando datos desde el directorio: {args.data_dir}")
    X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(args.data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(args.data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(args.data_dir, "y_test.csv")).values.ravel()

    # Filtrar las características según lo definido en la configuración
    features_used = config['features_used']
    logger.info(f"Usando {len(features_used)} características seleccionadas del experimento ganador.")
    X_train_final = X_train[features_used]
    X_test_final = X_test[features_used]

    # Obtener la instancia del modelo con sus mejores hiperparámetros
    model = get_model_instance(config['model_name'], config['parameters'])

    # Configurar MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(f"{config.get('name', 'house_price_model')}_Production")

    # Iniciar ejecución de MLflow
    with mlflow.start_run(run_name="final_model_training") as run:
        logger.info(f"Entrenando el modelo final: {config['model_name']} en el conjunto de entrenamiento.")
        model.fit(X_train_final, y_train)
        logger.info("Modelo entrenado exitosamente.")
        
        # Evaluar el modelo en el conjunto de prueba
        logger.info("Evaluando el modelo en el conjunto de prueba...")
        y_pred = model.predict(X_test_final)
        
        # Calcular métricas finales
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        logger.info(f"Métricas finales en test: R²={r2:.4f}, RMSE=${rmse:,.2f}, MAE=${mae:,.2f}")

        # Registrar en MLflow
        mlflow.log_params(config['parameters'])
        mlflow.log_metrics({'r2': r2, 'rmse': rmse, 'mae': mae})
        mlflow.log_param("features_used_count", len(features_used))
        
        # Log y registro del modelo en un solo paso
        model_name = config.get('name', 'house_price_model')
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train_final.head(5),
            registered_model_name=model_name
        )
        logger.info(f"Modelo '{model_name}' registrado en MLflow.")
        
        # Guardar el modelo final localmente
        save_dir = os.path.join(args.models_dir, "production")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "final_model.pkl")
        joblib.dump(model, save_path)
        
        logger.info(f"Modelo final guardado localmente en: {save_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

