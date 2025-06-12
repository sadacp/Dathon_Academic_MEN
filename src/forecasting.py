import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from prophet import Prophet
import os
import sys

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from settings import FIGURES_PATH # Ruta de figuras global

# Manejo de advertencias
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Realiza un pronóstico para múltiples indicadores utilizando el modelo Prophet de Facebook.
#Genera y muestra gráficos del pronóstico y sus componentes.

def forecast_with_prophet(df, indicators, target_year=2026, date_col='anio',
                          figsize_plot=(9, 6), figsize_components=(9, 6),
                          plot_components=True, save_dir=None):
    
    if df.empty or indicators is None or not indicators:
        print("⚠️ El DataFrame de entrada está vacío o la lista de indicadores está vacía. No se puede realizar el pronóstico.")
        return {}

    if date_col not in df.columns:
        print(f"⛔ Error: La columna de fecha '{date_col}' no se encuentra en el DataFrame.")
        return {}

    all_forecasts = {}

    for indicador in indicators:
        if indicador not in df.columns:
            print(f"⚠️ Advertencia: El indicador '{indicador}' no se encuentra en el DataFrame. Saltando.")
            continue

        print(f'\n📊 Modelo Prophet para: {indicador.upper()}')

        # Preparar datos: Renombrar columnas a 'ds' (fecha) y 'y' (valor)
        # Se toma una copia para evitar SettingWithCopyWarning
        df_prophet = df[[date_col, indicador]].copy()
        df_prophet.rename(columns={date_col: 'ds', indicador: 'y'}, inplace=True)
        
        # Convertir la columna 'ds' a formato de fecha (primer día del año)
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')

        # Calcular años a predecir hasta el target_year
        ultimo_anio = df_prophet['ds'].max().year
        if target_year <= ultimo_anio:
            print(f"⚠️ Advertencia: El año objetivo {target_year} es igual o anterior al último año en los datos ({ultimo_anio}). No se realizarán pronósticos futuros para {indicador}.")
            periods_to_predict = 0
        else:
            periods_to_predict = target_year - ultimo_anio

        # Crear y entrenar modelo Prophet
        # Silenciar la salida por defecto de Prophet si no hay datos o la salida no es necesaria
        # (se puede ajustar verbosidad en producción)
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        
        # Filtrar NaN en la columna 'y' antes de ajustar el modelo
        df_prophet_clean = df_prophet.dropna(subset=['y'])
        
        if df_prophet_clean.empty:
            print(f"⚠️ No hay datos válidos para el indicador '{indicador}' después de eliminar NaN. No se puede entrenar el modelo.")
            continue
        
        model.fit(df_prophet_clean)

        # Crear fechas futuras hasta el target_year
        future = model.make_future_dataframe(periods=periods_to_predict, freq='Y')
        forecast = model.predict(future)

        # Mostrar últimos valores pronosticados
        print("Últimos valores del pronóstico:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        all_forecasts[indicador] = forecast

        # Graficar pronóstico
        fig_forecast, ax_forecast = plt.subplots(figsize=figsize_plot)
        model.plot(forecast, ax=ax_forecast)
        ax_forecast.set_title(f'Pronóstico de {indicador.replace("_", " ").title()} (hasta {target_year})')
        ax_forecast.set_xlabel('Año')
        ax_forecast.set_ylabel(f'{indicador.replace("_", " ").title()}')
        ax_forecast.grid(True)
        plt.tight_layout()
        
        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                fig_forecast.savefig(os.path.join(save_dir, f'prophet_forecast_{indicador}.png'), bbox_inches='tight')
                print(f" ✅ Gráfico de pronóstico guardado en: {os.path.join(save_dir, f'prophet_forecast_{indicador}.png')}")
            except Exception as e:
                print(f"⛔ Error al guardar el gráfico de pronóstico para {indicador}: {e}")
        plt.show()
        plt.close(fig_forecast) # Cerrar la figura para liberar memoria

        # Graficar componentes del pronóstico (opcional)
        if plot_components:
            fig_components = model.plot_components(forecast)
            fig_components.set_size_inches(figsize_components) # Ajustar tamaño de componentes
            plt.tight_layout()
            
            if save_dir:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    # Prophet.plot_components devuelve una figura, no necesita ax=
                    fig_components.savefig(os.path.join(save_dir, f'prophet_components_{indicador}.png'), bbox_inches='tight')
                    print(f"✅ Gráfico de componentes guardado en: {os.path.join(save_dir, f'prophet_components_{indicador}.png')}")
                except Exception as e:
                    print(f"⛔ Error al guardar el gráfico de componentes para {indicador}: {e}")
            plt.show()
            plt.close(fig_components) # Cerrar la figura para liberar memoria

    print("\n---✅  Pronóstico con Prophet finalizado para todos los indicadores ---")
    return all_forecasts


#Realiza un pronóstico para múltiples indicadores utilizando el modelo ARIMA.
#Genera y muestra gráficos del pronóstico ARIMA.

def forecast_with_arima(df, indicators, target_year=2026, date_col='anio', 
                        arima_order=(1, 1, 1), figsize=(16, 10), save_dir=None):
    
    # Ignorar advertencias, especialmente de la convergencia de ARIMA
    warnings.filterwarnings("ignore")

    print("\n--- INICIANDO PRONÓSTICO CON MODELO ARIMA ---")

    if df.empty or indicators is None or not indicators:
        print("⚠️ El DataFrame de entrada está vacío o la lista de indicadores está vacía. No se puede realizar el pronóstico ARIMA.")
        warnings.filterwarnings("default") # Restablecer advertencias
        return {}

    if date_col not in df.columns:
        print(f"⛔ Error: La columna de fecha '{date_col}' no se encuentra en el DataFrame.")
        warnings.filterwarnings("default") # Restablecer advertencias
        return {}

    all_arima_forecasts = {}

    # Calcular el número de filas para los subplots (2 columnas)
    nrows = (len(indicators) + 1) // 2
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    axs = axs.flatten() # Aplanar el array de ejes para facilitar la iteración

    for i, indicador in enumerate(indicators):
        if indicador not in df.columns:
            print(f"⚠️ Advertencia: El indicador '{indicador}' no se encuentra en el DataFrame. Saltando.")
            if i < len(axs): # Si hay un subplot para este índice
                fig.delaxes(axs[i]) # Eliminar el subplot vacío
            continue

        print(f'📈 Modelo ARIMA para: {indicador.upper()}')

        # Preparar los datos de la serie de tiempo (promedio anual)
        df_ts = df[[date_col, indicador]].groupby(date_col).mean().reset_index()
        
        # Convertir la columna de año a formato de fecha y establecer como índice
        # Se añade YearEnd(0) para asegurar que el índice sea de fin de año si es necesario
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], format='%Y') + pd.offsets.YearEnd(0)
        df_ts = df_ts.sort_values(date_col)
        df_ts.set_index(date_col, inplace=True)
        df_ts.index.freq = 'YE' # Establecer la frecuencia a 'Year End'

        # Filtrar NaN en la serie antes de modelar
        ts_data = df_ts[indicador].dropna()

        if ts_data.empty:
            print(f"⚠️ No hay datos válidos para el indicador '{indicador}' después de eliminar NaN. No se puede entrenar el modelo ARIMA.")
            if i < len(axs):
                fig.delaxes(axs[i])
            continue
        
        if len(ts_data) < 2: # ARIMA necesita al menos 2 puntos de datos
            print(f"⚠️ No hay suficientes puntos de datos para el indicador '{indicador}' ({len(ts_data)}). Se requieren al menos 2. Saltando ARIMA.")
            if i < len(axs):
                fig.delaxes(axs[i])
            continue

        try:
            # Modelo ARIMA
            modelo = ARIMA(ts_data, order=arima_order)
            modelo_fit = modelo.fit()

            # Calcular el número de pasos para pronosticar
            last_year_in_data = ts_data.index[-1].year
            if target_year <= last_year_in_data:
                steps = 0
                print(f"⚠️ Advertencia: El año objetivo {target_year} es igual o anterior al último año en los datos ({last_year_in_data}). No se realizarán pronósticos futuros para {indicador}.")
            else:
                steps = target_year - last_year_in_data
            
            forecast = pd.Series() # Inicializar como serie vacía
            if steps > 0:
                forecast = modelo_fit.forecast(steps=steps)
                # Asegurarse de que el índice del pronóstico coincida con los años futuros esperados
                forecast.index = pd.date_range(start=ts_data.index[-1] + pd.offsets.YearEnd(1), periods=steps, freq='YE')
            
            all_arima_forecasts[indicador] = forecast

            # Graficar
            ax = axs[i]
            ax.plot(ts_data.index, ts_data, label='Histórico', color='#003366', linewidth=2.5)
            if not forecast.empty:
                ax.plot(forecast.index, forecast, label='Pronóstico ARIMA', linestyle='--', color='#FFA500', linewidth=2.5)

            ax.set_title(f'{indicador.replace("_", " ").title()} (ARIMA)', fontsize=12, weight='bold')
            ax.set_xlabel('Año', fontsize=10)
            ax.set_ylabel(f'{indicador.replace("_", " ").title()} (%)', fontsize=10)
            ax.legend(frameon=False, fontsize=9)

            # Estilizar bordes y ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_alpha(0.5)
            ax.spines['bottom'].set_alpha(0.5)
            ax.tick_params(axis='both', labelsize=9)
            ax.set_facecolor("white") # Fondo del subplot

        except Exception as e:
            print(f"⛔ Error al ajustar o pronosticar con ARIMA para '{indicador}': {e}")
            if i < len(axs):
                fig.delaxes(axs[i]) # Eliminar el subplot si hay un error
            all_arima_forecasts[indicador] = pd.Series(dtype='float64') # Registrar como serie vacía en caso de error
            continue

    # Ocultar subplots sobrantes si los hay
    # Si el número de indicadores válidos es impar, el último subplot queda vacío
    num_valid_indicators = len([ind for ind in indicators if ind in df.columns])
    if num_valid_indicators < len(axs):
        for j in range(num_valid_indicators, len(axs)):
            if axs[j] is not None: # Solo borrar si no ha sido borrado por un error de indicador
                fig.delaxes(axs[j])

    plt.tight_layout()
    plt.suptitle('Pronósticos ARIMA por Indicador', fontsize=16, weight='bold', y=1.04)
    plt.subplots_adjust(top=0.92) # Ajustar el espacio para el título general
    
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'arima_forecasts_overview.png'), bbox_inches='tight')
            print(f"✅ Gráfica general de pronósticos ARIMA guardada en: {os.path.join(save_dir, 'arima_forecasts_overview.png')}")
        except Exception as e:
            print(f"⛔ Error al guardar la gráfica general de pronósticos ARIMA: {e}")
    plt.show()
    plt.close(fig) # Cerrar la figura para liberar memoria

    print("\n--- Pronóstico con ARIMA finalizado para todos los indicadores ✅---")
    warnings.filterwarnings("default") # Restablecer advertencias a su comportamiento por defecto
    return all_arima_forecasts

# Realiza un pronóstico para múltiples indicadores utilizando un modelo de Regresión Lineal.
def forecast_with_linear_regression(df, indicators, target_years=[2024, 2025, 2026],
                                    date_col='anio', figsize=(14, 12), save_dir=None):
    
    print("\n--- INICIANDO PRONÓSTICO CON REGRESIÓN LINEAL ---")

    if df.empty or indicators is None or not indicators:
        print("⚠️ El DataFrame de entrada está vacío o la lista de indicadores está vacía. No se puede realizar el pronóstico con Regresión Lineal.")
        return pd.DataFrame()

    if date_col not in df.columns:
        print(f"⛔ Error: La columna de fecha '{date_col}' no se encuentra en el DataFrame.")
        return pd.DataFrame()

    # Preparar datos agrupados por año con promedio de cada variable
    # Seleccionar solo las columnas necesarias y hacer una copia explícita
    df_lr = df[[date_col] + indicators].copy()
    df_lr = df_lr.groupby(date_col).mean().reset_index()
    
    # Asegurar que no haya NaN en las columnas de indicadores después del agrupamiento,
    # ya que LinearRegression no los maneja.
    df_lr.dropna(subset=indicators, inplace=True)

    if df_lr.empty:
        print("⚠️ No hay datos válidos después de agrupar y eliminar NaN para los indicadores. No se puede realizar la regresión lineal.")
        return pd.DataFrame()

    # Año como número para regresión, escalando desde el año mínimo
    min_anio = df_lr[date_col].min()
    df_lr['anio_num'] = df_lr[date_col] - min_anio

    # Crear DataFrame para predicciones futuras
    future_nums = np.array(target_years) - min_anio
    predicciones = {date_col: target_years} # Usar date_col para la columna de años en el resultado

    # Configurar subplots (3 filas, 2 columnas como ejemplo, pero adaptable)
    # Calcular el número de filas necesario para la cuadrícula
    nrows = math.ceil(len(indicators) / 2)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    axs = axs.flatten() # Aplanar para facilitar la iteración

    for i, var in enumerate(indicators):
        if var not in df_lr.columns:
            print(f"⚠️ Advertencia: El indicador '{var}' no se encuentra en el DataFrame de datos agrupados. Saltando.")
            if i < len(axs):
                fig.delaxes(axs[i])
            continue
        
        # Asegurar que X y y sean DataFrames/Series no vacías
        X = df_lr[['anio_num']]
        y = df_lr[var]

        if X.empty or y.empty:
            print(f"⚠️ Advertencia: Datos insuficientes para el indicador '{var}'. Saltando regresión lineal.")
            if i < len(axs):
                fig.delaxes(axs[i])
            continue

        model_lr = LinearRegression()
        try:
            model_lr.fit(X, y)
            preds = model_lr.predict(future_nums.reshape(-1, 1))
            predicciones[f'{var}_predicha'] = preds
        except Exception as e:
            print(f"⛔ Error al ajustar el modelo de Regresión Lineal para '{var}': {e}")
            predicciones[f'{var}_predicha'] = [np.nan] * len(target_years) # Registrar NaN para este indicador
            if i < len(axs):
                fig.delaxes(axs[i]) # Eliminar el subplot si hay un error
            continue

        # Visualización
        ax = axs[i]
        ax.scatter(df_lr[date_col], df_lr[var], color='blue', label='Histórico')
        ax.plot(df_lr[date_col], model_lr.predict(X), color='green', label='Tendencia lineal')
        ax.plot(target_years, preds, 'o--', color='red', label=f'Pronóstico {target_years[0]}–{target_years[-1]}')
        
        ax.set_title(f'Regresión Lineal de {var.replace("_", " ").title()} hasta {target_years[-1]}')
        ax.set_xlabel('Año')
        ax.set_ylabel(var.replace("_", " ").title())
        ax.legend()
        ax.grid(True)

    # Ocultar subplots sobrantes si los hay
    num_plots_generated = len([var for var in indicators if var in df_lr.columns])
    if num_plots_generated < len(axs):
        for j in range(num_plots_generated, len(axs)):
            if axs[j] is not None: # Solo borrar si no ha sido borrado por un error de indicador
                fig.delaxes(axs[j])

    plt.tight_layout()
    plt.suptitle('Pronósticos por Regresión Lineal', fontsize=16, weight='bold', y=1.03) # Título general
    plt.subplots_adjust(top=0.9) # Ajustar para que el título no se superponga
    
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'linear_regression_forecasts_overview.png'), bbox_inches='tight')
            print(f"✅ Gráfica general de pronósticos de Regresión Lineal guardada en: {os.path.join(save_dir, 'linear_regression_forecasts_overview.png')}")
        except Exception as e:
            print(f"⛔ Error al guardar la gráfica general de pronósticos de Regresión Lineal: {e}")
    plt.show()
    plt.close(fig) # Cerrar la figura para liberar memoria

    # Mostrar predicciones en un DataFrame
    df_pred = pd.DataFrame(predicciones)
    df_pred = df_pred[[date_col] + [f'{var}_predicha' for var in indicators if f'{var}_predicha' in df_pred.columns]] # Reordenar columnas
    print("\n📈 Predicciones para años futuros:")
    print(df_pred.round(2))
    
    print("\n--- Pronóstico con Regresión Lineal finalizado ✅---")
    return df_pred

# Realiza un pronóstico para múltiples indicadores utilizando un modelo de Regresión Polinómica.
def forecast_with_polynomial_regression(df, indicators, target_years=[2024, 2025, 2026],
                                        date_col='anio', degree=2, figsize=(16, 14), save_dir=None):
    
    print(f"\n--- INICIANDO PRONÓSTICO CON REGRESIÓN POLINÓMICA (Grado {degree}) ---")

    if df.empty or indicators is None or not indicators:
        print("El DataFrame de entrada está vacío o la lista de indicadores está vacía. No se puede realizar el pronóstico con Regresión Polinómica.")
        return pd.DataFrame()

    if date_col not in df.columns:
        print(f"Error: La columna de fecha '{date_col}' no se encuentra en el DataFrame.")
        return pd.DataFrame()

    # Limpiar y preparar datos: agrupar por año y calcular el promedio
    df_poly = df[[date_col] + indicators].copy()
    df_poly = df_poly.groupby(date_col).mean().reset_index()
    
    # Eliminar filas donde hay NaN en cualquiera de los indicadores
    df_poly.dropna(subset=indicators, inplace=True)

    if df_poly.empty:
        print("No hay datos válidos después de agrupar y eliminar NaN para los indicadores. No se puede realizar la regresión polinómica.")
        return pd.DataFrame()

    # Año como número para regresión, escalando desde el año mínimo
    min_anio = df_poly[date_col].min()
    df_poly['anio_num'] = df_poly[date_col] - min_anio

    # Años futuros para predicción
    future_nums = np.array(target_years) - min_anio

    predicciones = {date_col: target_years}

    # Configurar subplots dinámicamente
    nrows = math.ceil(len(indicators) / 2)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    axs = axs.flatten() # Aplanar el array de ejes para facilitar la iteración

    for i, var in enumerate(indicators):
        if var not in df_poly.columns:
            print(f"Advertencia: El indicador '{var}' no se encuentra en el DataFrame de datos agrupados. Saltando.")
            if i < len(axs):
                fig.delaxes(axs[i])
            continue

        X = df_poly[['anio_num']]
        y = df_poly[var]

        if X.empty or y.empty:
            print(f"Advertencia: Datos insuficientes para el indicador '{var}'. Saltando regresión polinómica.")
            if i < len(axs):
                fig.delaxes(axs[i])
            continue
        
        # Asegurarse de que haya suficientes puntos para el grado del polinomio
        if len(X) < degree + 1:
            print(f"Advertencia: No hay suficientes puntos de datos ({len(X)}) para una regresión polinómica de grado {degree} para '{var}'. Saltando.")
            if i < len(axs):
                fig.delaxes(axs[i])
            predicciones[f'{var}_pred'] = [np.nan] * len(target_years)
            continue

        try:
            # Crear y entrenar el modelo de regresión polinómica
            model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model_poly.fit(X, y)

            # Realizar predicciones futuras
            preds = model_poly.predict(future_nums.reshape(-1, 1))
            predicciones[f'{var}_pred'] = preds
        except Exception as e:
            print(f"Error al ajustar el modelo de Regresión Polinómica para '{var}': {e}")
            predicciones[f'{var}_pred'] = [np.nan] * len(target_years)
            if i < len(axs):
                fig.delaxes(axs[i])
            continue


        # Graficar
        ax = axs[i]
        ax.scatter(df_poly[date_col], y, color='blue', label='Datos históricos')
        # Crear un rango de años para el ajuste polinómico suave
        anio_range = np.linspace(df_poly[date_col].min(), df_poly[date_col].max(), 100).reshape(-1, 1)
        anio_num_range = anio_range - min_anio
        ax.plot(anio_range, model_poly.predict(anio_num_range), color='orange', label=f'Ajuste polinómico grado {degree}')
        ax.plot(target_years, preds, 'o--', color='red', label=f'Pronóstico {target_years[0]}–{target_years[-1]}')
        
        ax.set_title(f'Regresión Polinómica de {var.replace("_", " ").title()}')
        ax.set_xlabel('Año')
        ax.set_ylabel(var.replace("_", " ").title())
        ax.legend()
        ax.grid(True)

    # Ocultar subplots sobrantes si los hay
    num_plots_generated = len([var for var in indicators if var in df_poly.columns and not df_poly[var].dropna().empty and len(df_poly[[var]].dropna()) >= degree + 1])
    if num_plots_generated < len(axs):
        for j in range(num_plots_generated, len(axs)):
            if axs[j] is not None:
                fig.delaxes(axs[j])

    plt.tight_layout()
    plt.suptitle(f'Pronósticos por Regresión Polinómica (Grado {degree})', fontsize=16, weight='bold', y=1.03)
    plt.subplots_adjust(top=0.9)
    
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'polynomial_regression_forecasts_degree_{degree}_overview.png'), bbox_inches='tight')
            print(f"Gráfica general de pronósticos de Regresión Polinómica guardada en: {os.path.join(save_dir, f'polynomial_regression_forecasts_degree_{degree}_overview.png')}")
        except Exception as e:
            print(f"Error al guardar la gráfica general de pronósticos de Regresión Polinómica: {e}")
    plt.show()
    plt.close(fig)

    # Crear DataFrame con predicciones futuras
    df_pred = pd.DataFrame(predicciones)
    df_pred = df_pred[[date_col] + [f'{var}_pred' for var in indicators if f'{var}_pred' in df_pred.columns]]
    print(f"\n📈 Predicciones polinómicas para {target_years[0]}–{target_years[-1]}:")
    print(df_pred.round(2))
    
    print("\n--- Pronóstico con Regresión Polinómica finalizado ---")
    return df_pred