import pandas as pd 
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# PARÁMETROS 
INPUT_CSV = '../../results/02_preparation/infrastructure/historical/if/05_variance.csv'       # CSV ORIGINAL
OUTPUT_CSV = '../../results/03_execution/01_classification/01_contaminated.csv'              # CSV CONTAMINADO
RESULTS_FOLDER = '../../results/02_preparation'                                               # CARPETA DE RESULTADOS

CONTAMINATION_RATE = 0.01
# PORCENTAJE DE FILAS ANÓMALAS (2%)
# DEFINE CUÁNTOS DATOS SERÁN CONSIDERADOS ANOMALÍAS
# MIN: 0.001, MAX: 0.1, RECOMENDADO: 0.01-0.05
# AFECTA SENSIBILIDAD Y DEFINICIÓN DE VALORES EXTREMOS

NOISE_INTENSITY = 3.0
# INTENSIDAD DEL RUIDO APLICADO A ANOMALÍAS
# CUANTO MAYOR, MAYORES DESVIOS RESPECTO A VALORES NORMALES
# MIN: 0.1, MAX: 5-10, RECOMENDADO: 2-4
# AFECTA MÁXIMOS, MÍNIMOS Y DETECTABILIDAD

OPERACION_RUIDO = 'SUMA'  
# OPERACIÓN PARA GENERAR RUIDO: 'SUMA', 'RESTA', 'MULTIPLICACION', 'ESCALA'
# DEFINE CÓMO SE MODIFICAN LOS VALORES ORIGINALES

ALL_COLUMNS = True             
# TRUE = CONTAMINAR TODAS LAS COLUMNAS NUMÉRICAS
COLUMNS_TO_CONTAMINATE = ['pressure', 'flow_rate']  
# COLUMNAS A CONTAMINAR SI ALL_COLUMNS = FALSE
CONTAMINATION_SCOPE = True      
# TRUE = TODAS COLUMNAS SELECCIONADAS, LISTA = COLUMNAS ESPECÍFICAS

ADD_LABEL = True               
# TRUE = AÑADIR COLUMNA 'IS_ANOMALY' PARA INDICAR FILAS CONTAMINADAS
RANDOM_STATE = 42              
# SEMILLA PARA ASEGURAR REPRODUCIBILIDAD
SHOW_INFO = True               
# TRUE = MOSTRAR INFORMACIÓN DE PROCESO EN CONSOLA

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)  # LEER CSV
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# SELECCIONAR COLUMNAS A CONTAMINAR
if ALL_COLUMNS:
    cols_to_contaminate = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  # TODAS COLUMNAS NUMÉRICAS
else:
    cols_to_contaminate = [c for c in COLUMNS_TO_CONTAMINATE if c in df.columns]  # FILTRAR COLUMNAS EXISTENTES

if isinstance(CONTAMINATION_SCOPE, list):
    cols_to_contaminate = [c for c in CONTAMINATION_SCOPE if c in df.columns]  # FILTRAR SEGÚN LISTA

ignored_cols = [c for c in (COLUMNS_TO_CONTAMINATE if not ALL_COLUMNS else []) if c not in df.columns]

if not cols_to_contaminate:
    raise ValueError("[ ERROR ] NO HAY COLUMNAS VÁLIDAS PARA CONTAMINAR")

if SHOW_INFO:
    print(f"[ INFO ] COLUMNAS A CONTAMINAR: {len(cols_to_contaminate)}")
    if ignored_cols:
        print(f"[ INFO ] COLUMNAS IGNORADAS: {ignored_cols}")

# SELECCIONAR FILAS A CONTAMINAR
np.random.seed(RANDOM_STATE)  # FIJAR SEMILLA
n_rows = df.shape[0]
n_contam = int(n_rows * CONTAMINATION_RATE)  # CALCULAR NÚMERO DE FILAS A CONTAMINAR
contam_indices = np.random.choice(df.index, size=n_contam, replace=False)  # INDICES ALEATORIOS
if SHOW_INFO:
    print(f"[ INFO ] FILAS A CONTAMINAR: {n_contam}")

# COPIAR DATASET ORIGINAL
df_contaminated = df.copy()  # TRABAJAR SOBRE COPIA

# APLICAR CONTAMINACIÓN CON OPERACIÓN CONFIGURABLE
for col in cols_to_contaminate:
    if df[col].dtype in ['int64', 'float64']:
        noise = np.random.normal(0, NOISE_INTENSITY * df[col].std(), size=n_contam)  # GENERAR RUIDO

        if OPERACION_RUIDO == 'SUMA':
            df_contaminated.loc[contam_indices, col] += noise  # SUMAR RUIDO
        elif OPERACION_RUIDO == 'RESTA':
            df_contaminated.loc[contam_indices, col] -= noise  # RESTAR RUIDO
        elif OPERACION_RUIDO == 'MULTIPLICACION':
            df_contaminated.loc[contam_indices, col] *= (1 + noise)  # ESCALAR VALOR ORIGINAL
        elif OPERACION_RUIDO == 'ESCALA':
            df_contaminated.loc[contam_indices, col] = df[col].mean() + noise  # MEDIA + RUIDO
        else:
            raise ValueError("[ ERROR ] OPERACION_RUIDO NO RECONOCIDA")

if SHOW_INFO:
    print(f"[ INFO ] CONTAMINACIÓN APLICADA CON OPERACIÓN '{OPERACION_RUIDO}'")

# AÑADIR COLUMNA DE ANOMALÍA
if ADD_LABEL:
    df_contaminated['is_anomaly'] = 0  # INICIALIZAR COMO NORMAL
    df_contaminated.loc[contam_indices, 'is_anomaly'] = 1  # MARCAR FILAS CONTAMINADAS
    if SHOW_INFO:
        print("[ INFO ] COLUMNA 'IS_ANOMALY' AÑADIDA")

# GUARDAR DATASET CONTAMINADO
df_contaminated.to_csv(OUTPUT_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] DATASET CONTAMINADO EN '{OUTPUT_CSV}'")
