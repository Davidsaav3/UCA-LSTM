import pandas as pd      
import numpy as np          
import os               
import sys
sys.stdout.reconfigure(encoding='utf-8')

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/03_diagnostic'  # CARPETA DE RESULTADOS
INPUT_IF_CSV = '../../results/03_execution/01_classification/01_if.csv'  # CSV IF
INPUT_LSTM_CSV = '../../results/03_execution/02_prediction/02_lstm_predictions.csv'  # CSV LSTM

OUTPUT_DIAGNOSTIC_CSV = os.path.join(RESULTS_FOLDER, '03_diagnostic.csv')             # CSV COMPLETO
OUTPUT_DIAGNOSTIC_DIF_CSV = os.path.join(RESULTS_FOLDER, '03_diagnostic_dif.csv')     # REGISTROS DIFERENTES DE "Correcto"
OUTPUT_DIAGNOSTIC_DEPURATED_CSV = os.path.join(RESULTS_FOLDER, '03_diagnostic_depurated.csv')  # SOLO REGISTROS "Correcto"

# PARÁMETROS DE DIAGNÓSTICO
THRESHOLD = 1.0  # UMBRAL PARA DIFERENCIA SIGNIFICATIVA

# FLAGS DE CONTROL
SHOW_INFO = True  # TRUE = IMPRIMIR INFO EN PANTALLA

# CARGAR DATOS
df_if = pd.read_csv(INPUT_IF_CSV)      # LEER CSV IF
df_lstm = pd.read_csv(INPUT_LSTM_CSV)  # LEER CSV PREDICCIONES LSTM

if SHOW_INFO:
    print(f"[ INFO ] DATOS IF CARGADOS: {df_if.shape}")      # MOSTRAR DIMENSIONES IF
    print(f"[ INFO ] DATOS LSTM CARGADOS: {df_lstm.shape}")  # MOSTRAR DIMENSIONES LSTM

# ASEGURAR MISMA LONGITUD ENTRE DATAFRAMES
min_len = min(len(df_if), len(df_lstm))
df_if = df_if.iloc[:min_len].copy()
df_lstm = df_lstm.iloc[:min_len].copy()

# RENOMBRAR COLUMNAS SI ES NECESARIO
if 'pred_step10' not in df_lstm.columns:
    df_lstm.rename(columns={df_lstm.columns[0]: 'pred_step10'}, inplace=True)  # RENOMBRAR COLUMNA PREDICCIÓN
if 'anomaly' not in df_if.columns:
    df_if.rename(columns={df_if.columns[0]: 'anomaly'}, inplace=True)          # RENOMBRAR COLUMNA ANOMALÍA

# 🔹 FUNCION DE DIAGNÓSTICO
def diagnostic(row):
    diff = abs(row['value'] - row['pred_step10'])  # CALCULAR DIFERENCIA ABSOLUTA
    if row['anomaly'] == 1:
        if diff >= THRESHOLD:
            return 'Confirmed'   # ANOMALÍA CONFIRMADA
        else:
            return 'Falso Positivo'  # ANOMALÍA INCORRECTA
    else:
        if diff >= THRESHOLD:
            return 'Falso Negativo'  # ANOMALÍA NO DETECTADA
        else:
            return 'Correcto'        # REGISTRO CORRECTO

# CREAR DATAFRAME UNIFICADO
df_diag = pd.DataFrame()
df_diag['value'] = df_if['wifi_inal_sf_1_39'] if 'wifi_inal_sf_1_39' in df_if.columns else df_if.iloc[:,1]  # VALORES REALES
df_diag['anomaly'] = df_if['anomaly']        # FLAG ANOMALÍA
df_diag['pred_step10'] = df_lstm['pred_step10']  # PREDICCIONES

# APLICAR DIAGNÓSTICO
df_diag['diagnostic'] = df_diag.apply(diagnostic, axis=1)  # DIAGNOSTICAR CADA FILA

# COLUMNA DE DIFERENCIA
df_diag['diff'] = abs(df_diag['value'] - df_diag['pred_step10'])  # DIFERENCIA ABSOLUTA

# GUARDAR CSV COMPLETO
df_diag.to_csv(OUTPUT_DIAGNOSTIC_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] DIAGNÓSTICO COMPLETO EN '{OUTPUT_DIAGNOSTIC_CSV}'")

# REGISTROS DIFERENTES DE "Correcto"
df_diag_dif = df_diag[df_diag['diagnostic'] != 'Correcto'].copy()  # FILTRAR DIFERENCIAS
df_diag_dif.to_csv(OUTPUT_DIAGNOSTIC_DIF_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] REGISTROS DIFERENTES DE 'Correcto' EN '{OUTPUT_DIAGNOSTIC_DIF_CSV}'")

# REGISTROS SOLO "Correcto" PARA DEPURACIÓN
df_diag_depurated = df_diag[df_diag['diagnostic'] == 'Correcto'].copy()  # FILTRAR SOLO CORRECTOS
df_diag_depurated.to_csv(OUTPUT_DIAGNOSTIC_DEPURATED_CSV, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] REGISTROS 'Correcto' EN '{OUTPUT_DIAGNOSTIC_DEPURATED_CSV}'")
