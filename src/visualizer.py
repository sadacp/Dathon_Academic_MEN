import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os
import sys

import plotly.graph_objects as go
import plotly.io as pio 

# Para K-Means
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Añade la ruta de la carpeta 'config' para poder importar settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))

from settings import FIGURES_PATH, DEFAULT_FIGURE_SIZE,SMALL_FIGURE_SIZE

# Guarda una figura en la ruta especificada
def save_figure(fig, filename, save_path=FIGURES_PATH):
    
    full_path = os.path.join(save_path, filename)
    try:
        fig.savefig(full_path, bbox_inches='tight')
        print(f"✅ Figura guardada en: {full_path}")
    except Exception as e:
        print(f"⛔ Error al guardar la figura en {full_path}: {e}")
    plt.close(fig) # Cierra la figura para liberar memoria

#Guarda una figura Plotly interactiva.
def save_plotly_figure(fig_plotly, filename, save_path=FIGURES_PATH, format='html'):
   
    full_path = os.path.join(save_path, filename)
    try:
        if format == 'html':
            pio.write_html(fig_plotly, full_path, auto_open=False)
        elif format in ['png', 'jpeg']:
            pio.write_image(fig_plotly, full_path, format=format)
        elif format == 'json':
            pio.write_json(fig_plotly, full_path)
        else:
            print(f"⚠️ Formato '{format}' no soportado para guardar figura Plotly. Use 'html', 'png', 'jpeg', 'json'.")
            return

        print(f"✅ Figura Plotly guardada como {format.upper()} en: {full_path}")
    except Exception as e:
        print(f"⛔ Error al guardar la figura Plotly como {format.upper()} en {full_path}: {e}")



# Genera histogramas para columnas numéricas en una cuadrícula organizada.
def plot_histograms(df, columns, bins=30, title_prefix="Distribución de ", 
                        save_path_prefix=None, cols=3, figsize_multiplier=5):
    
    if df is None:
        print("⛔ Error: DataFrame es None. No se pueden generar histogramas.")
        return
    
    # Filtrar solo columnas existentes y numéricas
    variables_existentes = [col for col in columns 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    n = len(variables_existentes)
    if n == 0:
        print("⚠️ Advertencia: No hay columnas numéricas válidas para graficar.")
        return
    
    # Configurar grid
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols*figsize_multiplier, rows*(figsize_multiplier-1)))
    
    print(f"📊 Generando histogramas para {n} columnas numéricas...")
    
    for i, var in enumerate(variables_existentes, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[var].dropna(), kde=True, bins=bins)
        plt.title(f"{title_prefix}{var.replace('_', ' ').title()}", fontsize=10)
        plt.xlabel(var.replace('_', ' ').title())
        plt.ylabel("Frecuencia")
        plt.grid(True)
    
    plt.tight_layout()
    
    # Guardar figura si se especificó
    if save_path_prefix is not None:
        save_path = f"{save_path_prefix}_histogram_grid.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico guardado como: {save_path}")
    
    plt.show()
    
    # Mostrar advertencias para columnas no válidas
    columnas_no_validas = set(columns) - set(variables_existentes)
    for col in columnas_no_validas:
        if col not in df.columns:
            print(f"⚠️ Advertencia: Columna '{col}' no encontrada en el DataFrame.")
        elif not pd.api.types.is_numeric_dtype(df[col]):
            print(f"⚠️ Advertencia: Columna '{col}' no es numérica. Saltando histograma.")

#Genera un gráfico de líneas interactivo de la evolución temporal de los indicadores claves.
def plotly_time_series(df, time_col, indicators, title='Evolución temporal de indicadores promedio',
                            yaxis_title='Valor promedio (%)', save_filename=None, width=800, height=450):
    
    if df is None or df.empty:
        print("⛔ Error: DataFrame es None o está vacío. No se puede generar gráfico de Plotly.")
        return
    if time_col not in df.columns:
        print(f"⛔ Error: Columna de tiempo '{time_col}' no encontrada en el DataFrame.")
        return
    
    # Crear la figura
    fig = go.Figure()

    # Añadir una línea por cada indicador
    for indicador in indicators:
        if indicador in df.columns:
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[f'{indicador}'],
                mode='lines+markers',
                name=indicador.replace('_', ' ').title() # Formatear el nombre para la leyenda
            ))
        else:
            print(f"⚠️ Advertencia: Indicador '{indicador}' no encontrado en el DataFrame. Saltando.")

    # Personalizar la figura
    fig.update_layout(
        title=title,
        xaxis_title=time_col.replace('_', ' ').title(),
        yaxis_title=yaxis_title,
        hovermode='x unified', # Muestra la información de todos los trazados en el punto del cursor
        template='plotly_white',
        width=width,
        height=height,
        # Asegurarse de que el eje X sea de tipo categórico o que los años se muestren correctamente
        xaxis=dict(tickmode='array', tickvals=df[time_col].unique())
    )
    
    # Mostrar o guardar la figura
    if save_filename:
        fig.show()
        save_plotly_figure(fig, save_filename, FIGURES_PATH, format='html') # Guarda como HTML por defecto
        
    else:
        fig.show()

# Genera y muestra múltiples gráficos de líneas interactivos (Plotly)
# para la evolución temporal de indicadores divididos por subcategorías (niveles educativos).

def plotly_subcategories_time_series(df, subcategories_dict, time_col,
                                          save_path_prefix=os.path.join(FIGURES_PATH, 'evolucion_nivel_')):
   
    if df is None or df.empty:
        print("⛔ Error: DataFrame es None o está vacío. No se pueden generar gráficos por subcategorías.")
        return
    if time_col not in df.columns:
        print(f"⛔Error: Columna de tiempo '{time_col}' no encontrada en el DataFrame.")
        return

    print(f"\n Generando gráficos de evolución temporal por subcategorías ({len(subcategories_dict)} indicadores principales)...")

    # Asegurarse de que el año sea numérico para el ordenamiento y tickvals
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df.dropna(subset=['anio'], inplace=True)
    
    # Obtener los años únicos para los ticks del eje X
    unique_years = sorted(df[time_col].unique())

    df_temp = df.groupby(time_col).mean(numeric_only=True).reset_index()

    # Crear un gráfico por cada indicador principal
    for indicador_principal, columnas_subcategorias in subcategories_dict.items():
        fig = go.Figure()

        # Añadir una línea por cada subcategoría (nivel)
        found_subcategories = []
        for col_subcat in columnas_subcategorias:
            if col_subcat in df.columns:
                nivel = col_subcat.split('_')[-1].capitalize()  # Extrae el nivel: primaria, secundaria, media
                fig.add_trace(go.Scatter(
                    x=df_temp[time_col],
                    y=df_temp[col_subcat],
                    mode='lines+markers',
                    name=nivel
                ))
                found_subcategories.append(col_subcat)
            else:
                print(f"Advertencia: Columna de subcategoría '{col_subcat}' no encontrada para '{indicador_principal}'. Saltando.")

        if not found_subcategories:
            print(f"Advertencia: No se encontraron columnas válidas para el indicador principal '{indicador_principal}'. Saltando gráfico.")
            continue # Pasa al siguiente indicador principal si no hay subcategorías válidas

        # Personalizar cada gráfico
        fig.update_layout(
            title=f'Evolución temporal de {indicador_principal.replace("_", " ").capitalize()} por nivel educativo',
            xaxis_title=time_col.replace('_', ' ').capitalize(),
            yaxis_title=f'{indicador_principal.replace("_", " ").capitalize()} promedio (%)',
            hovermode='x unified',
            template='plotly_white',
            width=850,
            height=450,
            legend_title='Nivel educativo',
            xaxis=dict(
                tickmode='array', # Muestra solo los años presentes
                tickvals=unique_years,
                dtick=1 # Asegura que cada año se muestre si es posible
            )
        )
        
        # Mostrar o guardar la figura
        if save_path_prefix:
            fig.show()
            filename = f"{save_path_prefix}{indicador_principal}.html"
            save_plotly_figure(fig, os.path.basename(filename), os.path.dirname(filename), format='html')
        else:
            fig.show()

# Genera múltiples heatmaps para mostrar la evolución temporal de subcategorías
def heatmap_sub_time_series(df, subcategories_dict, time_col='anio',
                                          title_suffix='por subcategoría',
                                          cmap='YlGnBu', fmt=".1f", linewidths=0.5,
                                          save_path_prefix=None, fig_size=(10, 4)):
    
    if df is None or df.empty:
        print("⛔ Error: DataFrame es None o está vacío. No se pueden generar heatmaps de subcategorías.")
        return
    if time_col not in df.columns:
        print(f"⛔ Error: Columna de tiempo '{time_col}' no encontrada en el DataFrame.")
        return

    print(f"\nGenerando heatmaps para subcategorías de indicadores por {time_col}...")

    # Asegurarse de que el año sea numérico para el ordenamiento
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df.dropna(subset=[time_col], inplace=True)


    for indicador_principal, columnas_subcategoria in subcategories_dict.items():
        # Crear una copia del sub-DataFrame para el heatmap
        # Filtrar solo las columnas que realmente existen en el DataFrame
        existing_subcategories = [col for col in columnas_subcategoria if col in df.columns]

        if not existing_subcategories:
            print(f"Advertencia: No se encontraron subcategorías válidas para '{indicador_principal}'. Saltando.")
            continue

        df_heatmap_data = df[[time_col] + existing_subcategories].copy()
        df_heatmap_data.set_index(time_col, inplace=True)

        # Renombrar columnas para visualización (solo la última parte: PRIMARIA, SECUNDARIA, MEDIA)
        # Ajusta la lógica de split si el formato de tus nombres de columna es diferente
        df_heatmap_data.columns = [
            col.split('_')[-1].capitalize() if '_' in col else col.capitalize()
            for col in df_heatmap_data.columns
        ]
        # Casos especiales para 'media' y 'transicion' si existen
        df_heatmap_data.rename(columns={'Media': 'Media', 'Transicion': 'Transición'}, inplace=True)


        fig, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(df_heatmap_data, annot=True, cmap=cmap, fmt=fmt, linewidths=linewidths, ax=ax)
        ax.set_title(f'Heatmap de {indicador_principal.replace("_", " ").capitalize()} {title_suffix}')
        ax.set_ylabel(time_col.replace('_', ' ').title())
        ax.set_xlabel('Subcategoría')
        plt.tight_layout()

        # Mostrar o guardar la figura
        if save_path_prefix:
            plt.show()
            # Generar un nombre de archivo único para cada indicador
            filename = f"heatmap_{indicador_principal}_por_nivel.png"
            save_figure(fig, filename, save_path=save_path_prefix)
        else:
            plt.show()

    print("✅ Generación de heatmaps de subcategorías completada.")

#Genera múltiples boxplots mostrando la distribución de indicadores a lo largo del tiempo (por año).
def plot_yearly_boxplots(df, indicators, titles_dict, time_col='anio',
                         nrows=None, ncols=2, figsize=(12, 10),
                         save_filename=None):
    
    if df is None or df.empty:
        print("⛔ Error: DataFrame es None o está vacío. No se pueden generar boxplots anuales.")
        return
    if time_col not in df.columns:
        print(f"⛔ Error: Columna de tiempo '{time_col}' no encontrada en el DataFrame.")
        return

    # Filtrar solo los indicadores que realmente existen en el DataFrame
    existing_indicators = [ind for ind in indicators if ind in df.columns]

    if not existing_indicators:
        print("⚠️ Advertencia: No se encontraron indicadores válidos para generar boxplots anuales. Saltando.")
        return

    # Calcular nrows automáticamente si no se especifica
    if nrows is None:
        nrows = math.ceil(len(existing_indicators) / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() # Aplanar el array de ejes para facilitar la iteración

    print(f"\nGenerando boxplots anuales para {len(existing_indicators)} indicadores...")

    # Generar cada boxplot en un subplot
    for i, indicador in enumerate(existing_indicators):
        ax = axes[i] # Asignar el eje actual
        sns.boxplot(data=df, x=time_col, y=indicador, ax=ax, legend=False) 
       
        
        # Obtener título del diccionario
        display_title = titles_dict.get(indicador, indicador.replace('_', ' ').capitalize())
        
        ax.set_title(f'Distribución de la {display_title.lower()} por año')
        ax.set_xlabel('Año')
        ax.set_ylabel(f'Valor {display_title} (%)' if 'porcentaje' in display_title.lower() or 'tasa' in display_title.lower() else f'{display_title} (Valor)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45) # Rotar etiquetas del eje X para mayor legibilidad

    # Si hay espacios vacíos en los subplots, eliminarlos
    if len(existing_indicators) < len(axes):
        for j in range(len(existing_indicators), len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout() # Ajustar el layout para evitar superposiciones

    # Mostrar o guardar la figura
    if save_filename:
        plt.show()
        save_figure(fig, save_filename, FIGURES_PATH) 
    else:
        plt.show()

    print("✅ Generación de boxplots anuales completada.")

# Genera un heatmap de la matriz de correlación Spearman

def plot_heatmap_correlation(df, indicators, time_col='anio',
                             title="Correlación de Spearman entre indicadores educativos (promedios anuales)",
                             method='spearman', cmap='YlGnBu', vmin=-1, vmax=1,
                             figsize=(8, 4), save_filename=None):
   
    if df is None or df.empty:
        print("Error: DataFrame es None o está vacío. No se puede generar el heatmap de correlación.")
        return
    if time_col not in df.columns:
        print(f"Error: Columna de tiempo '{time_col}' no encontrada en el DataFrame.")
        return

    # Filtrar solo los indicadores que realmente existen en el DataFrame
    existing_indicators = [ind for ind in indicators if ind in df.columns]

    if len(existing_indicators) < 2:
        print("⚠️ Advertencia: Se necesitan al menos dos indicadores válidos para calcular la correlación. Saltando.")
        return

    print(f"\nCalculando y generando heatmap de correlación para {len(existing_indicators)} indicadores...")

    # Agrupar por año y calcular el promedio de los indicadores existentes
    df_mean_by_year = df.groupby(time_col)[existing_indicators].mean(numeric_only=True)

    # Calcular matriz de correlación con el método especificado
    corr_matrix = df_mean_by_year.corr(method=method)

    # Visualizar el heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    # Mostrar o guardar la figura
    if save_filename:
        plt.show()
        save_figure(fig, save_filename, FIGURES_PATH)
    else:
        plt.show()

    print("Generación de heatmap de correlación completada.")


#Genera gráficos de distribución (histplot y kdeplot) para comparar indicadores
# entre dos años específicos.
def plot_distribution_comparison_two_years(df, indicators, year_start, year_end,
                                           time_col='anio', bins=30, alpha=0.6,
                                           color_start='blue', color_end='red',
                                           figsize_per_indicator=(12, 4), save_filename=None):
    
    if df is None or df.empty:
        print("⛔ Error: DataFrame es None o está vacío. No se pueden generar gráficos de distribución comparativos.")
        return
    if time_col not in df.columns:
        print(f"⛔ Error: Columna de tiempo '{time_col}' no encontrada en el DataFrame.")
        return

    
    # Filtrar solo los indicadores que realmente existen en el DataFrame
    existing_indicators = [ind for ind in indicators if ind in df.columns]

    if not existing_indicators:
        print("⚠️ Advertencia: No se encontraron indicadores válidos para generar gráficos de distribución comparativos. Saltando.")
        return

    # Calcular el tamaño total de la figura
    total_height = len(existing_indicators) * figsize_per_indicator[1]
    fig, axes = plt.subplots(nrows=len(existing_indicators), ncols=2,
                             figsize=(figsize_per_indicator[0], total_height))

    # Asegurarse de que axes sea un array 2D incluso para un solo indicador
    if len(existing_indicators) == 1:
        axes = np.array([axes])

    print(f"\nGenerando gráficos de distribución comparativos para {len(existing_indicators)} indicadores entre {year_start} y {year_end}...")

    for i, indicador in enumerate(existing_indicators):
        # Datos para el año inicial
        data_start_year = df[df[time_col] == year_start][indicador].dropna()
        # Datos para el año final
        data_end_year = df[df[time_col] == year_end][indicador].dropna()

        # --- Gráfico de distribución (Histplot) ---
        ax_hist = axes[i, 0]
        sns.histplot(data_start_year, kde=True, color=color_start,
                     label=f"Año {year_start}", alpha=alpha, bins=bins, ax=ax_hist)
        sns.histplot(data_end_year, kde=True, color=color_end,
                     label=f"Año {year_end}", alpha=alpha, bins=bins, ax=ax_hist)

        ax_hist.set_title(f"Distribución de {indicador.replace('_', ' ').title()} ({year_start} vs {year_end})")
        ax_hist.set_xlabel(f"Tasa de {indicador.replace('_', ' ').title()} (%)")
        ax_hist.set_ylabel("Frecuencia de Municipios")
        ax_hist.legend()
        ax_hist.grid(True, linestyle="--", alpha=0.7)

        # --- Gráfico de densidad (KDEplot) ---
        ax_kde = axes[i, 1]
        sns.kdeplot(data_start_year, color=color_start,
                    label=f"Año {year_start}", fill=True, alpha=alpha, ax=ax_kde)
        sns.kdeplot(data_end_year, color=color_end,
                    label=f"Año {year_end}", fill=True, alpha=alpha, ax=ax_kde)

        ax_kde.set_title(f"Densidad de {indicador.replace('_', ' ').title()} ({year_start} vs {year_end})")
        ax_kde.set_xlabel(f"Tasa de {indicador.replace('_', ' ').title()} (%)")
        ax_kde.set_ylabel("Densidad")
        ax_kde.legend()
        ax_kde.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout() # Ajustar el layout para evitar superposiciones

    # Mostrar o guardar la figura
    if save_filename:
        plt.show()
        save_figure(fig, save_filename, FIGURES_PATH) # Usar la función save_figure para Matplotlib
    else:
        plt.show()

    print("✅ Generación de gráficos de distribución comparativos completada.")


# Genera y muestra el gráfico del "Método del Codo" para determinar el número óptimo
# de clusters (K) para el algoritmo K-Means.

def plot_kmeans_elbow(dataframe, max_clusters=10, random_state=42):
   
    if dataframe.empty:
        print("El DataFrame de entrada está vacío. No se puede generar el gráfico del método del codo.")
        return

    inertias = []
    # Rango de posibles números de clusters
    for i in range(1, max_clusters + 1):
        kmeans_test = KMeans(n_clusters=i, random_state=random_state, n_init=10) # n_init para reproducibilidad
        kmeans_test.fit(dataframe)
        inertias.append(kmeans_test.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.title('Método del Codo para K-Means: Inercia vs. Número de Clusters (K)')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    plt.show()

#Aplica PCA a los datos escalados y genera una visualización de los clusters.
def view_clusters_pca(df_scaled, etiquetas_cluster, titulo="Clustering de municipios según indicadores educativos", 
                           tamaño_figura=(10, 4), colormap='viridis', tamaño_puntos=50, 
                           transparencia=0.7, mostrar_grid=True, save_filename=None):
   
    # Aplicar PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df_scaled)
    
    # Crear visualización
    plt.figure(figsize=tamaño_figura)
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=etiquetas_cluster, 
                         cmap=colormap, s=tamaño_puntos, alpha=transparencia)
    
    plt.title(titulo)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar(scatter, label="Cluster")
    
    if mostrar_grid:
        plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()
    
    print("\nVisualización PCA de los clusters generada.")

#Genera gráficos de barras que muestran los promedios de indicadores
# para cada clúster, organizados en subplots.

def plot_cluster_bar_charts(df_clustered, indicators_list, cluster_col='Cluster', 
                            title_prefix='Promedio de indicadores por Cluster',
                            nrows=2, ncols=3, figsize=(18, 10), palette='viridis',
                            save_dir=None, filename='cluster_bar_charts.png'):
    
    if df_clustered.empty or indicators_list is None or not indicators_list:
        print("⚠️ El DataFrame de entrada está vacío o la lista de indicadores está vacía. No se puede generar el gráfico de barras de clusters.")
        return
    
    if cluster_col not in df_clustered.columns:
        print(f"⛔ Error: La columna de clúster '{cluster_col}' no se encuentra en el DataFrame.")
        return

    # Filtrar indicadores que realmente existen en el DataFrame
    valid_indicators = [ind for ind in indicators_list if ind in df_clustered.columns]
    if not valid_indicators:
        print("⚠️ Ninguno de los indicadores proporcionados se encontró en el DataFrame. No se generará el gráfico.")
        return

    print(f"\n--- {title_prefix} ---")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() # Aplanar el array de ejes para iterar fácilmente

    for i, indicador in enumerate(valid_indicators):
        if i < len(axes): # Asegurarse de no exceder el número de subplots
            sns.barplot(
                x=cluster_col,
                y=indicador,
                data=df_clustered,
                hue=cluster_col, # Usar 'Cluster' para diferenciar barras por color
                palette=palette,
                legend=False, # La leyenda individual de hue no es necesaria aquí
                ax=axes[i]
            )
            axes[i].set_title(f'Promedio de {indicador.replace("_", " ").title()}')
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel('Valor Promedio')
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        else:
            print(f"⚠️ Advertencia: No hay suficientes subplots para el indicador '{indicador}'. Considere aumentar nrows/ncols.")
            break # Salir del bucle si ya no hay subplots disponibles

    # Si hay subplots extra, los ocultamos
    for j in range(len(valid_indicators), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_dir:
        try:
            full_save_path = os.path.join(save_dir, filename)
            # Asegurarse de que el directorio exista
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(full_save_path, bbox_inches='tight')
            print(f"✅ Gráfico de barras de clusters guardado en: {full_save_path}")
        except Exception as e:
            print(f"⛔ Error al guardar el gráfico de barras de clusters en {full_save_path}: {e}")
    
    plt.show()

    print("\n✅ Gráfico de barras de promedios por cluster generado.")

#Genera boxplots que muestran la distribución de indicadores para cada clúster

def plot_cluster_box_plots(df_clustered, indicators_list, cluster_col='Cluster', 
                           title_prefix='Distribución de Indicadores por Cluster',
                           nrows=2, ncols=3, figsize=(18, 10), palette='viridis',
                           save_dir=None, filename='cluster_box_plots.png'):
    
    if df_clustered.empty or indicators_list is None or not indicators_list:
        print("⚠️ El DataFrame de entrada está vacío o la lista de indicadores está vacía. No se puede generar el gráfico de boxplots de clusters.")
        return
    
    if cluster_col not in df_clustered.columns:
        print(f"⛔ Error: La columna de clúster '{cluster_col}' no se encuentra en el DataFrame.")
        return

    # Filtrar indicadores que realmente existen en el DataFrame
    valid_indicators = [ind for ind in indicators_list if ind in df_clustered.columns]
    if not valid_indicators:
        print("⚠️ Ninguno de los indicadores proporcionados se encontró en el DataFrame. No se generará el gráfico.")
        return

    print(f"\n--- {title_prefix} ---")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() # Aplanar el array de ejes para iterar fácilmente

    for i, indicador in enumerate(valid_indicators):
        if i < len(axes): # Asegurarse de no exceder el número de subplots
            sns.boxplot(
                x=cluster_col,
                y=indicador,
                data=df_clustered,
                hue=cluster_col, # Usar 'Cluster' para diferenciar boxplots por color
                palette=palette,
                legend=False, # La leyenda individual de hue no es necesaria aquí
                ax=axes[i]
            )
            axes[i].set_title(f'Distribución de {indicador.replace("_", " ").title()}')
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel('Valor')
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        else:
            print(f"⚠️ Advertencia: No hay suficientes subplots para el indicador '{indicador}'. Considere aumentar nrows/ncols.")
            break # Salir del bucle si ya no hay subplots disponibles

    # Si hay subplots extra, los ocultamos
    for j in range(len(valid_indicators), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_dir:
        try:
            full_save_path = os.path.join(save_dir, filename)
            # Asegurarse de que el directorio exista
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(full_save_path, bbox_inches='tight')
            print(f"✅ Gráfico de boxplots de clusters guardado en: {full_save_path}")
        except Exception as e:
            print(f"⛔ Error al guardar el gráfico de boxplots de clusters en {full_save_path}: {e}")
    
    plt.show()

    print("\n✅ Gráfico de boxplots de distribución por cluster generado.")