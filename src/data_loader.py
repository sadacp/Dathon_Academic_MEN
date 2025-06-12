import pandas as pd
import geopandas as gpd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))

# Trae las rutas guardadas en Settings
from settings import RAW_DATA_PATH, PROCESSED_DATA_PATH, DATA_ENCODING, RAW_GJON_PATH

#Carga del dataset
def load_data(file_path=RAW_DATA_PATH, encoding=DATA_ENCODING):
    
    print(f"Intentando cargar el dataset desde: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"Dataset cargado exitosamente. Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en la ruta especificada: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: El archivo {file_path} está vacío.")
        return None
    except Exception as e:
        print(f"Ocurrió un error al cargar el archivo {file_path}: {e}")
        return None

def save_processed_data(df, file_path=PROCESSED_DATA_PATH):
    
    if df is None:
        print("Error: No se puede guardar un DataFrame nulo.")
        return

    try:
        df.to_csv(file_path)
        print(f"Datos procesados guardados exitosamente en: {file_path}")
    except Exception as e:
        print(f"Error al guardar los datos procesados en {file_path}: {e}")

#Carga el DataFrame procesado desde un archivo CSV.
def load_processed_data(file_path=PROCESSED_DATA_PATH):
    
    print(f"Intentando cargar los datos procesados desde: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Datos procesados cargados exitosamente. Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo procesado no se encontró en la ruta especificada: {file_path}")
        return None
    except Exception as e:
        print(f"Ocurrió un error al cargar el archivo procesado {file_path}: {e}")
        return None

#Carga un archivo GeoJSON en un GeoDataFrame de Geopandas.
def load_geojson(filepath=RAW_GJON_PATH, driver=None):
    
    if not os.path.exists(filepath):
        print(f"Error: El archivo GeoJSON no se encontró en la ruta: {filepath}")
        return None
    
    try:
        if driver:
            gdf = gpd.read_file(filepath, driver=driver)
        else:
            gdf = gpd.read_file(filepath)
        
        print(f"✅ Archivo GeoJSON cargado exitosamente desde: {filepath}")
        print(f"Columnas del GeoDataFrame: {gdf.columns.tolist()}")
        print(f"Número de geometrías: {len(gdf)}")
        return gdf
    except Exception as e:
        print(f"⛔ Error al cargar el archivo GeoJSON desde {filepath}: {e}")
        return None


if __name__ == "__main__":
    print("--- Probando funciones de data_loader.py ---")

    # Prueba de carga de datos brutos
    df_raw_test = load_data()
    if df_raw_test is not None:
        print("\nHead de datos brutos cargados:")
        print(df_raw_test.head())
        print("-" * 30)

        # Prueba de guardado de datos procesados (usando df_raw_test como ejemplo)
        # En un escenario real, aquí df_raw_test sería tu df_work_clean
        temp_processed_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'temp_test_processed.pkl')
        save_processed_data(df_raw_test, temp_processed_path)
        print("-" * 30)

        # Prueba de carga de datos procesados
        df_loaded_processed_test = load_processed_data(temp_processed_path)
        if df_loaded_processed_test is not None:
            print("\nHead de datos procesados cargados (prueba):")
            print(df_loaded_processed_test.head())
        print("-" * 30)

        # Limpiar archivo temporal de prueba
        if os.path.exists(temp_processed_path):
            os.remove(temp_processed_path)
            print(f"Archivo temporal de prueba eliminado: {temp_processed_path}")
    else:
        print("No se pudo cargar el archivo RAW para las pruebas.")