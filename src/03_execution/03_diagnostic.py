import pandas as pd           # PARA MANEJO DE DATAFRAMES
import numpy as np            # PARA MANIPULACIÓN NUMÉRICA
import os

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/03_diagnostic'                                    # CARPETA PRINCIPAL DE RESULTADOS

INPUT_IF_CSV = os.path.join(RESULTS_FOLDER, '../../results/03_execution/01_classification/01_if.csv')          # CSV CON ANOMALÍAS DE ISOLATION FOREST
INPUT_LSTM_CSV = os.path.join(RESULTS_FOLDER, '../../results/03_execution/02_prediction/02_lstm_predictions.csv')  # CSV CON PREDICCIONES LSTM
OUTPUT_DIAGNOSTIC_CSV = os.path.join(RESULTS_FOLDER, '03_diagnostic.csv') # CSV CON DIAGNÓSTICO

# PARÁMETROS DE DIAGNÓSTICO
THRESHOLD = 1.0        # UMBRAL PARA DIFERENCIA SIGNIFICATIVA (Diff(x,x̂) > Δ)

# FLAGS DE CONTROL
SHOW_INFO = True         # MOSTRAR MENSAJES INFORMATIVOS

# CARGAR DATOS
df_if = pd.read_csv(INPUT_IF_CSV)        # LEER CSV DE ANOMALÍAS IF
df_lstm = pd.read_csv(INPUT_LSTM_CSV)    # LEER CSV DE PREDICCIONES LSTM

if SHOW_INFO:
    print(f"[ INFO ] DATOS IF CARGADOS: {df_if.shape}")           # INFO DE DIMENSIONES IF
    print(f"[ INFO ] DATOS LSTM CARGADOS: {df_lstm.shape}")       # INFO DE DIMENSIONES LSTM

# ASEGURAR QUE LOS DATAFRAMES TIENEN MISMA LONGITUD
min_len = min(len(df_if), len(df_lstm))
df_if = df_if.iloc[:min_len].copy()         # RECORTAR IF SI ES NECESARIO
df_lstm = df_lstm.iloc[:min_len].copy()     # RECORTAR LSTM SI ES NECESARIO

# RENOMBRAR COLUMNAS SI ES NECESARIO
if 'prediction' not in df_lstm.columns:
    df_lstm.rename(columns={df_lstm.columns[0]: 'prediction'}, inplace=True)  # RENOMBRAR PREDICCIÓN
if 'anomaly' not in df_if.columns:
    df_if.rename(columns={df_if.columns[0]: 'anomaly'}, inplace=True)        # RENOMBRAR ANOMALÍA

# 🔹 DIAGNÓSTICO: DETERMINAR CATEGORÍA SEGÚN IF Y LSTM
def diagnostic(row):
    diff = abs(row['value'] - row['prediction'])  # DIFERENCIA ENTRE VALOR REAL Y PREDICCIÓN
    if row['anomaly'] == 1:
        if diff >= THRESHOLD:
            return 'Confirmed'      # ANOMALÍA CONFIRMADA (VP)
        else:
            return 'Falso Positivo' # FALSO POSITIVO (FP)
    else:
        if diff >= THRESHOLD:
            return 'Falso Negativo' # FALSO NEGATIVO (FN)
        else:
            return 'Correcto'       # REGISTRO CORRECTO

# CREAR DATAFRAME UNIFICADO PARA DIAGNÓSTICO
df_diag = pd.DataFrame()
df_diag['value'] = df_if['value'] if 'value' in df_if.columns else df_if.iloc[:,1]  # VALOR REAL
df_diag['anomaly'] = df_if['anomaly']                                              # INDICADOR IF
df_diag['prediction'] = df_lstm['prediction']                                       # PREDICCIÓN LSTM

# APLICAR FUNCION DE DIAGNÓSTICO
df_diag['diagnostic'] = df_diag.apply(diagnostic, axis=1)

# 🔹 CREAR COLUMNA DE DIFERENCIA PARA AUDITORÍA
df_diag['diff'] = abs(df_diag['value'] - df_diag['prediction'])

# GUARDAR RESULTADO
df_diag.to_csv(OUTPUT_DIAGNOSTIC_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] DIAGNÓSTICO COMPLETO EN '{OUTPUT_DIAGNOSTIC_CSV}'")
