import pandas as pd           # PARA MANEJO DE DATAFRAMES
import numpy as np            # PARA MANIPULACIÓN NUMÉRICA
import os

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/03_diagnostic'
INPUT_IF_CSV = '../../results/03_execution/01_classification/01_if.csv'
INPUT_LSTM_CSV = '../../results/03_execution/02_prediction/02_lstm_predictions.csv'

OUTPUT_DIAGNOSTIC_CSV = os.path.join(RESULTS_FOLDER, '03_diagnostic.csv')             # CSV COMPLETO
OUTPUT_DIAGNOSTIC_DIF_CSV = os.path.join(RESULTS_FOLDER, '03_diagnostic_dif.csv')     # DIFERENTES DE "Correcto"
OUTPUT_DIAGNOSTIC_DEPURATED_CSV = os.path.join(RESULTS_FOLDER, '03_diagnostic_depurated.csv')  # SOLO "Correcto"

# PARÁMETROS DE DIAGNÓSTICO
THRESHOLD = 1.0  # UMBRAL PARA DIFERENCIA SIGNIFICATIVA

# FLAGS DE CONTROL
SHOW_INFO = True

# CARGAR DATOS
df_if = pd.read_csv(INPUT_IF_CSV)
df_lstm = pd.read_csv(INPUT_LSTM_CSV)

if SHOW_INFO:
    print(f"[ INFO ] DATOS IF CARGADOS: {df_if.shape}")
    print(f"[ INFO ] DATOS LSTM CARGADOS: {df_lstm.shape}")

# ASEGURAR MISMA LONGITUD
min_len = min(len(df_if), len(df_lstm))
df_if = df_if.iloc[:min_len].copy()
df_lstm = df_lstm.iloc[:min_len].copy()

# RENOMBRAR COLUMNAS SI ES NECESARIO
if 'prediction' not in df_lstm.columns:
    df_lstm.rename(columns={df_lstm.columns[0]: 'prediction'}, inplace=True)
if 'anomaly' not in df_if.columns:
    df_if.rename(columns={df_if.columns[0]: 'anomaly'}, inplace=True)

# 🔹 DIAGNÓSTICO
def diagnostic(row):
    diff = abs(row['value'] - row['prediction'])
    if row['anomaly'] == 1:
        if diff >= THRESHOLD:
            return 'Confirmed'
        else:
            return 'Falso Positivo'
    else:
        if diff >= THRESHOLD:
            return 'Falso Negativo'
        else:
            return 'Correcto'

# CREAR DATAFRAME UNIFICADO
df_diag = pd.DataFrame()
df_diag['value'] = df_if['agua_map07020001'] if 'agua_map07020001' in df_if.columns else df_if.iloc[:,1]
df_diag['anomaly'] = df_if['anomaly']
df_diag['prediction'] = df_lstm['prediction']

# APLICAR DIAGNÓSTICO
df_diag['diagnostic'] = df_diag.apply(diagnostic, axis=1)

# COLUMNA DE DIFERENCIA
df_diag['diff'] = abs(df_diag['value'] - df_diag['prediction'])

# GUARDAR CSV COMPLETO
df_diag.to_csv(OUTPUT_DIAGNOSTIC_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] DIAGNÓSTICO COMPLETO EN '{OUTPUT_DIAGNOSTIC_CSV}'")

# REGISTROS DIFERENTES DE "Correcto"
df_diag_dif = df_diag[df_diag['diagnostic'] != 'Correcto'].copy()
df_diag_dif.to_csv(OUTPUT_DIAGNOSTIC_DIF_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] REGISTROS DIFERENTES DE 'Correcto' EN '{OUTPUT_DIAGNOSTIC_DIF_CSV}'")

# REGISTROS SOLO "Correcto" PARA DEPURACIÓN
df_diag_depurated = df_diag[df_diag['diagnostic'] == 'Correcto'].copy()
df_diag_depurated.to_csv(OUTPUT_DIAGNOSTIC_DEPURATED_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] REGISTROS 'Correcto' EN '{OUTPUT_DIAGNOSTIC_DEPURATED_CSV}'")
