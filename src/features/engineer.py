# src/features/engineer.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')

def create_features(df):
    """Crea nuevas características a partir de los datos existentes."""
    logger.info("Creando nuevas características")
    
    # Hacer una copia para evitar modificar el dataframe original
    df_featured = df.copy()
    
    # Calcular la antigüedad de la casa
    current_year = datetime.now().year
    df_featured['house_age'] = current_year - df_featured['year_built']
    logger.info("Característica 'house_age' creada")
    
    # Precio por pie cuadrado
    df_featured['price_per_sqft'] = df_featured['price'] / df_featured['sqft']
    logger.info("Característica 'price_per_sqft' creada")
    
    # Relación dormitorios/baños
    df_featured['bed_bath_ratio'] = df_featured['bedrooms'] / df_featured['bathrooms']
    # Manejar división por cero
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    logger.info("Característica 'bed_bath_ratio' creada")
    
    # NO hacer one-hot encoding aquí; dejar que el preprocesador lo maneje
    return df_featured

def create_preprocessor():
    """Crea un pipeline de preprocesamiento."""
    logger.info("Creando pipeline de preprocesamiento")
    
    # Definir grupos de características
    categorical_features = ['location', 'condition']
    numerical_features = ['sqft', 'bedrooms', 'bathrooms', 'house_age', 'price_per_sqft', 'bed_bath_ratio']
    
    # Preprocesamiento para características numéricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    # Preprocesamiento para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinar preprocesadores en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def run_feature_engineering(input_file, output_file, preprocessor_file):
    """Pipeline completo de ingeniería de características."""
    # Cargar datos limpios
    logger.info(f"Cargando datos desde {input_file}")
    df = pd.read_csv(input_file)
    
    # Crear nuevas características
    df_featured = create_features(df)
    logger.info(f"Dataset con características creado con forma: {df_featured.shape}")
    
    # Crear y ajustar el preprocesador
    preprocessor = create_preprocessor()
    X = df_featured.drop(columns=['price'], errors='ignore')  # Solo características
    y = df_featured['price'] if 'price' in df_featured.columns else None  # Columna objetivo (si existe)
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Preprocesador ajustado y características transformadas")
    
    # Guardar el preprocesador
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Preprocesador guardado en {preprocessor_file}")
    
    # Guardar los datos completamente preprocesados
    df_transformed = pd.DataFrame(X_transformed)
    if y is not None:
        df_transformed['price'] = y.values
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Datos completamente preprocesados guardados en {output_file}")
    
    return df_transformed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingeniería de características para datos de viviendas.')
    parser.add_argument('--input', required=True, help='Ruta al archivo CSV limpio')
    parser.add_argument('--output', required=True, help='Ruta para el archivo CSV de salida (características)')
    parser.add_argument('--preprocessor', required=True, help='Ruta para guardar el preprocesador')
    
    args = parser.parse_args()
    
    run_feature_engineering(args.input, args.output, args.preprocessor)
