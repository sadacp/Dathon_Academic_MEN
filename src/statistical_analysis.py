import pandas as pd
from scipy import stats
import numpy as np # Por si se necesita para operaciones con arrays, aunque stats.f_oneway ya maneja Series/arrays

#Realiza un Análisis de Varianza (ANOVA) para validar si existen diferencias
# estadísticamente significativas en los promedios de los indicadores entre los diferentes clústeres

def perform_cluster_anova(df_clustered, indicators_list, cluster_col='Cluster', alpha=0.05):
    
    print("\n--- INICIANDO ANÁLISIS ANOVA PARA VALIDAR BRECHAS ENTRE CLUSTERS ---")

    if df_clustered.empty or indicators_list is None or not indicators_list:
        print("⚠️ El DataFrame de entrada está vacío o la lista de indicadores está vacía. No se puede realizar el análisis ANOVA.")
        return None
    
    if cluster_col not in df_clustered.columns:
        print(f"⛔ Error: La columna de clúster '{cluster_col}' no se encuentra en el DataFrame.")
        return None

    results = []

    for ind in indicators_list:
        if ind not in df_clustered.columns:
            print(f"⚠️ Advertencia: El indicador '{ind}' no se encuentra en el DataFrame. Será omitido.")
            continue

        # Prepara los datos para ANOVA: 
        groups = []
        # Itera sobre los IDs de los clusters únicos y ordenados
        for cluster_id in sorted(df_clustered[cluster_col].unique()):
            # Selecciona los datos del indicador actual para los municipios de este cluster
            group_data = df_clustered[df_clustered[cluster_col] == cluster_id][ind].dropna()

            # Asegura que el grupo sea válido para ANOVA (al menos 2 puntos de datos)
            if not group_data.empty and len(group_data) > 1:
                groups.append(group_data)
            else:
                print(f"Advertencia: El Cluster {cluster_id} tiene pocos o ningún dato válido para '{ind}'. Será omitido en ANOVA.")

        # Realiza la prueba ANOVA solo si hay al menos dos grupos válidos para comparar
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups) # *groups desempaqueta la lista en argumentos individuales

            conclusion = ""
            if p_value < alpha:
                conclusion = "RECHAZAMOS la hipótesis nula (H0). Existen diferencias estadísticamente significativas."
            else:
                conclusion = "NO RECHAZAMOS la hipótesis nula (H0). No hay evidencia de diferencias significativas."
            
            print(f"\n--- Resultados ANOVA para: {ind.replace('_', ' ').title()} ---")
            print(f"  Estadístico F: {f_stat:.3f}")
            print(f"  P-valor: {p_value:.5f}")
            print(f"  **Conclusión:** Con un P-valor de {p_value:.5f} (que es {'menor' if p_value < alpha else 'mayor'} que {alpha}),")
            print(f"  {conclusion}")
            if p_value < alpha:
                print(f"  Esto respalda fuertemente la existencia de brechas territoriales en {ind.replace('_', ' ').lower()}.")
            else:
                print(f"  Esto sugiere que no hay brechas claras en {ind.replace('_', ' ').lower()} entre los clusters, o la muestra no es suficiente para detectarlas.")
            
            results.append({
                'Indicador': ind.replace('_', ' ').title(),
                'Estadístico F': f_stat,
                'P-valor': p_value,
                'Conclusión': conclusion
            })
        else:
            print(f"\n--- No se pudo realizar ANOVA para {ind.replace('_', ' ').title()} ---")
            print("  Razón: No hay suficientes grupos válidos o datos para comparar.")
            results.append({
                'Indicador': ind.replace('_', ' ').title(),
                'Estadístico F': np.nan,
                'P-valor': np.nan,
                'Conclusión': 'No se pudo realizar ANOVA (pocos datos/grupos válidos)'
            })

    print("\n--- ANÁLISIS ANOVA FINALIZADO ✅---")
    print("Estos p-valores son cruciales para validar las brechas territoriales identificadas.")
    
    return pd.DataFrame(results)


#Realiza un Análisis de Varianza (ANOVA) para validar si existen diferencias
# estadísticamente significativas en los promedios de los indicadores
# entre diferentes periodos temporales.

def perform_temporal_anova(df, indicators_list, date_col='anio', 
                           period_breaks=None, period_labels=None, 
                           period_col_name='periodo', alpha=0.05):
    
    print("\n--- INICIANDO ANÁLISIS ANOVA PARA VALIDAR BRECHAS ENTRE PERIODOS TEMPORALES ---")

    if df.empty or indicators_list is None or not indicators_list:
        print("⚠️ El DataFrame de entrada está vacío o la lista de indicadores está vacía. No se puede realizar el análisis ANOVA.")
        return pd.DataFrame() # Retorna DataFrame vacío en lugar de None para consistencia
    
    if date_col not in df.columns:
        print(f"⛔ Error: La columna de fecha '{date_col}' no se encuentra en el DataFrame.")
        return pd.DataFrame()

    df_copy = df.copy() # Trabajar con una copia para no modificar el DataFrame original

    # Definir la función de clasificación de períodos
    if period_breaks is None and period_labels is None:
        # Usar la clasificación por defecto del código original
        def clasificar_periodo_default(anio):
            if anio <= 2014:
                return 'Periodo 1 (2011-2014)'
            elif anio <= 2018:
                return 'Periodo 2 (2015-2018)'
            else: # anio > 2018
                return 'Periodo 3 (2019-2023)'
        df_copy[period_col_name] = df_copy[date_col].apply(clasificar_periodo_default)
    elif period_breaks is not None and period_labels is not None:
        if len(period_labels) != len(period_breaks) + 1:
            print("Error: La longitud de 'period_labels' debe ser len(period_breaks) + 1.")
            return pd.DataFrame()
        
        def clasificar_periodo_custom(anio):
            for i, break_year in enumerate(period_breaks):
                if anio <= break_year:
                    return period_labels[i]
            return period_labels[-1] # Para el último período (años mayores al último break)
        df_copy[period_col_name] = df_copy[date_col].apply(clasificar_periodo_custom)
    else:
        print("⛔ Error: Debe proporcionar 'period_breaks' y 'period_labels' juntos, o ninguno para usar los valores por defecto.")
        return pd.DataFrame()

    results = []

    # Iterar sobre cada variable
    for var in indicators_list:
        if var not in df_copy.columns:
            print(f"⚠️ Advertencia: El indicador '{var}' no se encuentra en el DataFrame. Será omitido.")
            # Añadir una entrada para el indicador no encontrado con NaN
            results.append({
                'Indicador': var.replace('_', ' ').title(),
                'Estadístico F': np.nan,
                'P-valor': np.nan,
                'Conclusión': 'Indicador no encontrado en el DataFrame'
            })
            continue

        grupos = []
        # Obtener los nombres de los períodos únicos y ordenados para asegurar consistencia
        unique_periods = sorted(df_copy[period_col_name].unique())
        
        for periodo in unique_periods:
            datos_periodo = df_copy[df_copy[period_col_name] == periodo][var].dropna()

            if not datos_periodo.empty and len(datos_periodo) > 1:
                grupos.append(datos_periodo)
            else:
                print(f"⚠️ Advertencia: El {periodo} tiene pocos o ningún dato válido para '{var}'. Será omitido en ANOVA.")

        # Realizar ANOVA si hay al menos 2 periodos con datos válidos
        if len(grupos) > 1:
            f_stat, p_value = stats.f_oneway(*grupos)

            conclusion = ""
            if p_value < alpha:
                conclusion = "RECHAZAMOS la hipótesis nula (H0). Existen diferencias estadísticamente significativas."
            else:
                conclusion = "NO RECHAZAMOS la hipótesis nula (H0). No hay evidencia de diferencias significativas."

            print(f"\n--- Resultados ANOVA para: {var.replace('_', ' ').title()} ---")
            print(f"  Estadístico F: {f_stat:.3f}")
            print(f"  P-valor: {p_value:.5f}")
            print(f"  **Conclusión:** Con un P-valor de {p_value:.5f} (que es {'menor' if p_value < alpha else 'mayor'} que {alpha}),")
            print(f"  {conclusion}")
            if p_value < alpha:
                print(f"  Esto respalda fuertemente la existencia de brechas temporales en este indicador.")
            else:
                print(f"  Esto sugiere que no hay brechas claras en {var.replace('_', ' ').lower()} entre los períodos, o la muestra no es suficiente para detectarlas.")
            
            results.append({
                'Indicador': var.replace('_', ' ').title(),
                'Estadístico F': f_stat,
                'P-valor': p_value,
                'Conclusión': conclusion
            })
        else:
            print(f"\n--- No se pudo realizar ANOVA para {var.replace('_', ' ').title()} ---")
            print("  Razón: No hay suficientes grupos válidos o datos para comparar.")
            results.append({
                'Indicador': var.replace('_', ' ').title(),
                'Estadístico F': np.nan,
                'P-valor': np.nan,
                'Conclusión': 'No se pudo realizar ANOVA (pocos datos/grupos válidos)'
            })

    print("\n--- ANÁLISIS ANOVA TEMPORAL FINALIZADO ✅ ---")
    print("Estos p-valores son cruciales para validar las brechas temporales identificadas.")
    
    return pd.DataFrame(results)
