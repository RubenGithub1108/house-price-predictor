# src/data/processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-processor')

def load_data(file_path):
    """Carga datos desde un archivo CSV."""
    logger.info(f"Cargando datos desde {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """Limpia el dataset manejando valores faltantes y outliers."""
    logger.info("Limpiando el dataset")
    
    # Hacer una copia para evitar modificar el DataFrame original
    df_cleaned = df.copy()
    
    # Manejo de valores faltantes
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()
        if missing_count > 0:
            logger.info(f"Se encontraron {missing_count} valores faltantes en {column}")
            
            # Para columnas numéricas, rellenar con la mediana
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                median_value = df_cleaned[column].median()
                df_cleaned[column] = df_cleaned[column].fillna(median_value)
                logger.info(f"Valores faltantes en {column} rellenados con la mediana: {median_value}")
            # Para columnas categóricas, rellenar con la moda
            else:
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                logger.info(f"Valores faltantes en {column} rellenados con la moda: {mode_value}")
    
    # Manejo de outliers en price (variable objetivo)
    # Usando el método IQR para identificar outliers
    Q1 = df_cleaned['price'].quantile(0.25)
    Q3 = df_cleaned['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filtrar outliers extremos
    outliers = df_cleaned[(df_cleaned['price'] < lower_bound) | 
                          (df_cleaned['price'] > upper_bound)]
    
    if not outliers.empty:
        logger.info(f"Se encontraron {len(outliers)} outliers en la columna price")
        df_cleaned = df_cleaned[(df_cleaned['price'] >= lower_bound) & 
                                (df_cleaned['price'] <= upper_bound)]
        logger.info(f"Outliers eliminados. Nueva forma del dataset: {df_cleaned.shape}")
    
    return df_cleaned

def process_data(input_file, output_file):
    """Pipeline completo de procesamiento de datos."""
    # Crear el directorio de salida si no existe
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    df = load_data(input_file)
    logger.info(f"Datos cargados con forma: {df.shape}")
    
    # Limpiar datos
    df_cleaned = clean_data(df)
    
    # Guardar datos procesados
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Datos procesados guardados en {output_file}")
    
    return df_cleaned

if __name__ == "__main__":
    # ejemplo de uso
    process_data(
        input_file="data/raw/house_data.csv", 
        output_file="data/processed/cleaned_house_data.csv"
    )