import pandas as pd
import unicodedata
from sklearn.cluster import KMeans

#Imprime un resumen general de un DataFrame, incluyendo sus dimensiones, primeras filas, información general y estadísticas descriptivas.
def summarize_dataframe(df, df_name="DataFrame"):
    
    if df is None:
        print(f"Error: No se puede resumir {df_name} porque es None.")
        return

    filas, columnas = df.shape
    print(f"El {df_name} tiene {filas} filas y {columnas} columnas.\n")

    print(f"Primeras filas del {df_name}:")
    print(df.head(), "\n")

    print(f"Información general del {df_name}:")
    df.info()
    print("\n") # Salto de línea después de info()

    print(f"Estadísticas descriptivas básicas del {df_name}:")
    print(df.describe(include='all'), "\n")


#Análisis de Valore Nulos
def analyze_completeness(df, df_name="DataFrame"):

    if df is None:
        print("Error: No se puede guardar un DataFrame nulo.")
        return
    
    total_posibles = df.size
    total_no_nulos = df.notnull().sum().sum()
    porcentaje_completo = (total_no_nulos / total_posibles) * 100

    print(f"Total de valores posibles: {total_posibles}")
    print(f"Total de valores no nulos: {total_no_nulos}")
    print(f"Porcentaje de completitud del DataFrame: {porcentaje_completo:.2f}%\n")

    porcentaje_nulos = df.isnull().mean().sort_values(ascending=False) * 100
    print("Porcentaje de valores nulos por columna (orden descendente):")
    print(porcentaje_nulos, "\n")

#Análisis de Columnas con Ceros :Cuenta cuántos ceros hay en cada columna numérica del DataFrame,
# y muestra las columnas que contienen al menos un cero

def count_zeros_by_column(df):

    if df is None:
        print("Error: No se puede guardar un DataFrame nulo.")
        return
    
    # Contar cuántos ceros hay por columna
    zero_counts = (df == 0).sum()

    # Filtrar columnas con al menos un cero
    columns_with_zeros = zero_counts[zero_counts > 0]

    # Mostrar resultados
    print(f"Número de columnas con al menos un valor en cero: {len(columns_with_zeros)}")
    print("\nColumnas con valores en cero y su cantidad:")
    print(columns_with_zeros)

    return columns_with_zeros

#Encuentra y muestra las columnas con datos duplicados en un DataFrame.
def find_duplicate_columns(df):
    
    columnas_duplicadas = []
    
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                columnas_duplicadas.append((df.columns[i], df.columns[j]))
                print(f"Las columnas '{df.columns[i]}' y '{df.columns[j]}' tienen los mismos datos.")
    
    if not columnas_duplicadas:
        print("No se encontraron columnas con datos duplicados.")
    
    return columnas_duplicadas

#Calcula estadísticas temporales 
def temporal_statistics(df, indicadores_claves, grupo_temporal='anio'):
    
    # Verificar que existan las columnas en el DataFrame
    indicadores_existentes = [col for col in indicadores_claves if col in df.columns]
    
    if not indicadores_existentes:
        raise ValueError("⛔ Ninguno de los indicadores existe en el DataFrame")
    
    # Verificar que exista la columna de agrupación
    if grupo_temporal not in df.columns:
        raise ValueError(f"⛔ La columna de agrupación '{grupo_temporal}' no existe")
    
    # Calcular estadísticas
    estadisticas = df.groupby(grupo_temporal)[indicadores_existentes].agg(['mean', 'median', 'std'])
    
    # Aplanar nombres de columnas
    estadisticas.columns = [f"{col[0]}_{col[1]}" for col in estadisticas.columns]
    
    # Resetear índice para tener la columna temporal como columna normal
    estadisticas.reset_index(inplace=True)
    
    # Información sobre el proceso
    print(f"✅ Estadísticas calculadas para {len(indicadores_existentes)}/{len(indicadores_claves)} indicadores")
    if len(indicadores_existentes) < len(indicadores_claves):
        faltantes = set(indicadores_claves) - set(indicadores_existentes)
        print(f"⚠️ Indicadores no encontrados: {faltantes}")
    
    return estadisticas

## Aplicar K-Means con el número óptimo de clusters 
def apply_kmeans_clustering(df_cluster_data, df_scaled, optimal_k):
    
    print(f"\nNúmero óptimo de clusters elegido (según Método del Codo): {optimal_k}")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)
    
    df_con_clusters = df_cluster_data.copy()
    df_con_clusters['Cluster'] = cluster_labels
    df_con_clusters = df_con_clusters.reset_index()
    
    print(f"\n✅ Clustering K-Means completado y etiquetas asignadas a {optimal_k} clusters.")
    print("Conteo de municipios por cluster:\n", df_con_clusters['Cluster'].value_counts())
    
    return df_con_clusters, cluster_labels
