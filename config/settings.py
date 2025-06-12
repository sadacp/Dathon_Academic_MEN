import os

# --- Rutas de Archivos y Directorios ---
# Obtiene la ruta base del proyecto (un nivel arriba de 'config')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas de datos
RAW_DATA_FILENAME = 'MEN_ESTADISTICAS_EN_EDUCACION.csv' # Nombre del archivo original
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', RAW_DATA_FILENAME)

RAW_GJON_FILENAME = 'colombia_departamentos.geojson.json' # Nombre del archivo original
RAW_GJON_PATH = os.path.join(BASE_DIR, 'data', 'raw', RAW_GJON_FILENAME)

PROCESSED_DATA_FILENAME = 'datos_educacion_limpios.csv' # Nombre del archivo de datos procesados
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', PROCESSED_DATA_FILENAME)

# Ruta para guardar modelos (si aplicara en el futuro)
MODELS_PATH = os.path.join(BASE_DIR, 'models')

# Ruta para guardar figuras/gráficos
FIGURES_PATH = os.path.join(BASE_DIR, 'reports', 'figuras')

# Asegurarse de que los directorios existan
os.makedirs(os.path.join(BASE_DIR, 'data', 'raw'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)


# --- Parámetros de Preprocesamiento y Análisis ---

# Columnas numéricas clave que parecen ser indicadores importantes

INDICADORES_CLAVE = [
    'tasa_matriculacion_5_16', 
    'cobertura_neta', 'cobertura_neta_primaria',  
    'cobertura_neta_secundaria', 'cobertura_neta_media', 
    'cobertura_bruta', 'cobertura_bruta_primaria',  
    'cobertura_bruta_secundaria', 'cobertura_bruta_media', 
    'desercion', 'desercion_primaria',  
    'desercion_secundaria', 'desercion_media', 
    'aprobacion', 'aprobacion_primaria',  
    'aprobacion_secundaria', 'aprobacion_media', 
    'reprobacion', 'reprobacion_primaria', 
    'reprobacion_secundaria', 'reprobacion_media',
    'repitencia', 'repitencia_primaria', 
    'repitencia_secundaria', 'repitencia_media'
]

INDICADORES_ESTADISTICO = [ 
    'cobertura_neta', 'cobertura_neta_primaria', 
    'cobertura_neta_secundaria', 'cobertura_neta_media', 
    'desercion', 'desercion_primaria', 
    'desercion_secundaria', 'desercion_media', 
    'aprobacion',  'aprobacion_primaria', 
    'aprobacion_secundaria', 'aprobacion_media', 
    'repitencia',  'repitencia_primaria', 
    'repitencia_secundaria', 'repitencia_media',
    'reprobacion',  'reprobacion_primaria','reprobacion_secundaria',
    'reprobacion_media'
]

INDICADORES_CLAVES_TASAS = [
    'cobertura_neta', 'cobertura_neta_primaria',
    'cobertura_neta_secundaria', 'cobertura_neta_media',
    'desercion', 'desercion_primaria',
    'desercion_secundaria', 'desercion_media',
    'aprobacion', 'aprobacion_primaria',
    'aprobacion_secundaria', 'aprobacion_media',
    'repitencia', 'repitencia_primaria',
    'repitencia_secundaria', 'repitencia_media',
    'reprobacion', 'reprobacion_primaria',
    'reprobacion_secundaria', 'reprobacion_media'
]

INDICADORES_PRINCIPALES_VIZ = [
    'cobertura_neta',
    'desercion',
    'aprobacion',
    'repitencia',
    'reprobacion',
    'poblacion_5_16'
]

INDICADORES_PLOTLY = [
    'cobertura_neta_mean', 'desercion_mean', 
    'aprobacion_mean', 'repitencia_mean', 'reprobacion_mean'
]

INDICADORES_BOXPLOTS = [
    'cobertura_neta', 'desercion', 
    'aprobacion', 'repitencia', 'reprobacion'
]

TITULOS_BOXPLOTS= {
    'cobertura_neta': 'Cobertura Neta',
    'desercion': 'Deserción',
    'aprobacion': 'Aprobación',
    'repitencia': 'Repitencia',
    'reprobacion': 'Reprobación'
}

SUBCATEGORIAS_PLOTLY = {
    'cobertura_neta': [
        'cobertura_neta_primaria',
        'cobertura_neta_secundaria', 
        'cobertura_neta_media'
    ],
    'desercion': [
        'desercion_primaria',
        'desercion_secundaria', 
        'desercion_media'
    ],
    'aprobacion': [
        'aprobacion_primaria',
        'aprobacion_secundaria', 
        'aprobacion_media'
    ],
    'repitencia': [
        'repitencia_primaria',
        'repitencia_secundaria', 
        'repitencia_media'
    ],
    'reprobacion': [
        'reprobacion_primaria',
        'reprobacion_secundaria', 
        'reprobacion_media'
    ]
}

# Columnas que deberían ser numéricas pero podrían tener valores no numéricos
COLUMNS_TO_CONVERT_TO_NUMERIC = [
    'poblacion_5_16' # También numérica por su uso en el modelo
]

# Columnas a eliminar 
COLUMNS_TO_DELETE = [
    'reprobacion_transicion',
    'aprobacion_transicion',
    'repitencia_transicion',
    'cobertura_neta_transicion',
    'cobertura_bruta_transicion',
    'desercion_transicion',
    'cobertura_bruta_transicion',
    'tamanio_promedio_de_grupo', 
    'sedes_conectadas_a_internet' 
]

# --- Columna a Filtrar
FILTER_COLUMN = 'municipio'

# --- Valor Filtrado
FILTER_VALUE = 'NACIONAL'

# --- Encoding para la carga de datos ---
DATA_ENCODING = 'utf-8' 


# --- Configuración de Visualizaciones ---
DEFAULT_FIGURE_SIZE = (10, 6)
SMALL_FIGURE_SIZE = (5, 4)
# Puedes añadir más configuraciones para colores, estilos, etc.