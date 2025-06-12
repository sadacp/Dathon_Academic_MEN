import pandas as pd
import numpy as np
import unicodedata
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))

# Limpia y estandariza los nombres de las columnas de un DataFrame.
# Convierte a minúsculas, reemplaza espacios por guiones bajos y elimina caracteres no alfanuméricos.

def clean_column_names(df):
    
    df_cleaned = df.copy()
    new_columns = []
    for col in df_cleaned.columns:
        # Convertir a minúsculas, reemplazar espacios por guiones bajos
        col = col.lower().replace(' ', '_').strip()
        col = col.replace('año', 'anio')  # Reemplazar antes de eliminar tildes
        # Eliminar acentos y caracteres especiales usando unicodedata
        col = unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('utf-8')
        # Eliminar cualquier caracter que no sea alfanumérico o guion bajo
        col = ''.join(c for c in col if c.isalnum() or c == '_')
        new_columns.append(col)
    df_cleaned.columns = new_columns
    print("Columnas estandarizadas ✅.")
  
    return df_cleaned

# Elimina columnas especificadas de un DataFrame y retorna información sobre el resultado

def delete_columns(df, columnas_a_eliminar):
   
    # Verificar qué columnas existen realmente en el DataFrame
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    
    # Crear una copia del DataFrame para no modificar el original
    df_nuevo = df.copy()
    
    # Eliminar las columnas
    df_nuevo.drop(columns=columnas_existentes, inplace=True)

    print("Columnas Eliminadas. ✅" )      
    return df_nuevo

#Imputa ceros y valores nulos en columnas numéricas con la mediana

def handle_missing_values(df):
    
    # Crear una copia del DataFrame para no modificar el original
    df_imp = df.copy()

    # Selecciona solo las columnas numéricas
    columnas_numericas = df_imp.select_dtypes(include='number').columns

    # Recorre cada columna numérica
    for col in columnas_numericas:
        # Calcular la mediana excluyendo ceros y nulos
        mediana = df_imp.loc[(df_imp[col] != 0) & (~df_imp[col].isnull()), col].median()
        
        # Reemplazar ceros por la mediana
        df_imp[col] = df_imp[col].replace(0, mediana)
        
        # Reemplazar nulos por la mediana
        df_imp[col] = df_imp[col].fillna(mediana)
    
    # Calcular totales
    total_ceros = (df_imp[col] == 0).sum().sum()
    total_nulos = df_imp[col].isnull().sum().sum()

    # Verificar imputaciones
    print("✅ Imputación completa.")
    print("Número de ceros por columna después de imputar: ", total_ceros , "\n")
    print("Número de nulos por columna después de imputar: ",total_nulos)
    
    
    return df_imp 

#Convierte columnas específicas a numérico, verifica nulos en categóricas e imputa valores si es necesario.
def normalize_column_names(df, columns_to_convert_to_numeric, imputar=True):
    
    # Hacer copia para no modificar el original
    df_procesado = df.copy()
    
    # 1. Verificar nulos en columnas categóricas
    columnas_categoricas = df_procesado.select_dtypes(include=['object']).columns
    nulos_categoricos = df_procesado[columnas_categoricas].isnull().sum()
    
    print("\n Valores nulos en columnas categóricas:")
    print(nulos_categoricos[nulos_categoricos > 0].to_string(), "\n")
    
    # 2. Conversión de columnas específicas a numérico
    for col in columns_to_convert_to_numeric:
        if col in df_procesado.columns:
            # Convertir a numérico (los errores se convierten a NaN)
            df_procesado[col] = pd.to_numeric(df_procesado[col], errors='coerce')
            
            # Imputar nulos con mediana si se solicita
            if imputar:
                mediana = df_procesado[col].median()
                df_procesado[col] = df_procesado[col].fillna(mediana)
                
                print(f"✅ Columna '{col}' convertida a numérico")
                print(f"   - Valores nulos imputados: {df[col].isnull().sum()} → 0")
                print(f"   - Mediana usada para imputación: {mediana:.2f}")
                print(f"   - Tipo de dato actual: {df_procesado[col].dtype}\n")
            else:
                print(f"✅ Columna '{col}' convertida a numérico (sin imputación)")
                print(f"   - Valores nulos actuales: {df_procesado[col].isnull().sum()}")
                print(f"   - Tipo de dato actual: {df_procesado[col].dtype}\n")
        else:
            print(f"⚠️ Advertencia: La columna '{col}' no existe en el DataFrame")
    
    return df_procesado

#Filtra un DataFrame eliminando las filas que contengan un valor específico en una columna dada.

def filter_remove_rows(df, columna, valor):
    
    # Verificar si la columna existe en el DataFrame
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame")
    
    # Filtrar el DataFrame eliminando las filas con el valor especificado
    df_filtrado = df[~(df[columna] == valor)].copy()
    
    # Información sobre el filtrado aplicado
    filas_eliminadas = len(df) - len(df_filtrado)
    print(f"✅ Se eliminaron {filas_eliminadas} filas donde '{columna}' == {valor}")
    
    return df_filtrado

