import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/01_classification'                       # CARPETA PRINCIPAL DE RESULTADOS
INPUT_CSV = '../../results/02_preparation/infrastructure/historical/if/05_variance.csv'     
OUTPUT_CSV = os.path.join(RESULTS_FOLDER, '01_if.csv')  
OUTPUT_IF_CSV = os.path.join(RESULTS_FOLDER, '01_if_anomaly.csv')   

# FLAGS DE CONTROL
SAVE_ANOMALY_CSV = True          # GUARDAR CSV SOLO CON ANOMALÍAS
SORT_ANOMALY_SCORE = True        # ORDENAR CSV DE ANOMALÍAS POR SCORE
INCLUDE_SCORE = True             # INCLUIR COLUMNA 'anomaly_score'
NORMALIZE_SCORE = True           # NORMALIZAR SCORE ENTRE 0 Y 1

# HIPERPARÁMETROS ISOLATION FOREST
N_ESTIMATORS = 100               # NÚMERO DE ÁRBOLES EN EL BOSQUE
MAX_SAMPLES = 'auto'             # MUESTRAS POR ÁRBOL
CONTAMINATION = 0.01             # FRACCIÓN ESTIMADA DE ANOMALÍAS
MAX_FEATURES = 1.0               # PROPORCIÓN DE FEATURES POR ÁRBOL
BOOTSTRAP = False                 # TRUE=MUESTREO CON REPETICIÓN
N_JOBS = -1                       # USAR TODOS LOS HILOS
RANDOM_STATE = 42                 # SEMILLA PARA REPRODUCIBILIDAD
VERBOSE = 0                        # NIVEL DE VERBOSIDAD
SHOW_INFO = True                   # MOSTRAR INFORMACIÓN

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV)       # LEER CSV DE DATOS DE INFRAESTRUCTURA
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# SEPARAR COLUMNA DE VERDAD
if 'is_anomaly' in df.columns:
    df_input = df.drop(columns=['is_anomaly'])  # NO USAR EN ENTRENAMIENTO
    is_anomaly_column = df['is_anomaly']       # GUARDAR PARA COMPARACIÓN
else:
    df_input = df.copy()
    is_anomaly_column = pd.Series([0]*len(df_input), name='is_anomaly')  # COLUMNA TEMPORAL

# SELECCIONAR COLUMNAS NUMÉRICAS
num_cols = df_input.select_dtypes(include=['int64', 'float64']).columns.tolist()  # SOLO FEATURES NUMÉRICAS

# ESCALAR DATOS
scaler = StandardScaler()              # ESCALADO PARA EVITAR DOMINANCIA DE MAGNITUDES
df_scaled = scaler.fit_transform(df_input[num_cols])  # TRANSFORMAR DATAFRAME A MATRIZ ESCALADA

# ENTRENAR ISOLATION FOREST
clf = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples=MAX_SAMPLES,
    contamination=CONTAMINATION,
    max_features=MAX_FEATURES,
    bootstrap=BOOTSTRAP,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    verbose=VERBOSE
)
clf.fit(df_scaled)                     # AJUSTAR MODELO A DATOS ESCALADOS

# CALCULAR SCORE Y PREDICCIÓN
anomaly_score = clf.decision_function(df_scaled) * -1  # MÁS POSITIVO = MÁS ANÓMALO
pred = clf.predict(df_scaled)                           # 1=NORMAL, -1=ANOMALÍA
df['anomaly'] = np.where(pred == 1, 0, 1)              # 0=normal, 1=anomalía
df['anomaly_score'] = anomaly_score                   # AÑADIR SCORE DE ANOMALÍA
df['is_anomaly'] = is_anomaly_column                  # REINSERTAR COLUMNA ORIGINAL

# INFORMACIÓN GENERAL
num_anomalies = df['anomaly'].sum()
num_normals = df.shape[0] - num_anomalies
if SHOW_INFO:
    print(f"[ INFO ] REGISTROS TOTALES: {df.shape[0]}")
    print(f"[ INFO ] ANOMALÍAS DETECTADAS: {num_anomalies}")
    print(f"[ INFO ] REGISTROS NORMALES: {num_normals}")
    print(f"[ INFO ] PORCENTAJE ANOMALÍAS: {num_anomalies/df.shape[0]*100:.2f}%")

# GUARDAR CSV COMPLETO
df.to_csv(OUTPUT_CSV, index=False)   # GUARDAR CSV CON TODAS LAS FILAS
if SHOW_INFO:
    print(f"[ GUARDADO ] CSV COMPLETO CON ANOMALÍAS EN '{OUTPUT_CSV}'")

# GUARDAR CSV SOLO CON ANOMALÍAS
if SAVE_ANOMALY_CSV:
    df_anomalies = df.loc[df['anomaly'] == 1].copy()  # FILTRAR SOLO ANOMALÍAS
    df_anomalies['anomaly_score'] = df_anomalies['anomaly_score'].astype(float)

    # NORMALIZAR SCORE ENTRE 0 Y 1
    if NORMALIZE_SCORE:
        min_score = df_anomalies['anomaly_score'].min()
        max_score = df_anomalies['anomaly_score'].max()
        if max_score > min_score:
            df_anomalies['anomaly_score'] = (df_anomalies['anomaly_score'] - min_score) / (max_score - min_score)

    # ORDENAR DE MAYOR A MENOR SEGÚN SCORE
    if SORT_ANOMALY_SCORE:
        df_anomalies = df_anomalies.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)

    # ELIMINAR SCORE SI NO SE DESEA INCLUIR
    if not INCLUDE_SCORE:
        df_anomalies.drop(columns=['anomaly_score'], inplace=True)

    # GUARDAR CSV FINAL SOLO ANOMALÍAS
    df_anomalies.to_csv(OUTPUT_IF_CSV, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] CSV ANOMALÍAS {'ORDENADAS' if SORT_ANOMALY_SCORE else ''} EN '{OUTPUT_IF_CSV}'")
