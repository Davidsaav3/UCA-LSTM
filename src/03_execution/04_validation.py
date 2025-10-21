import pandas as pd
import numpy as np
import os

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/04_supervision'
INPUT_IF_CSV = '../../results/03_execution/01_classification/01_if.csv'
INPUT_LSTM_CSV = '../../results/03_execution/02_prediction/02_lstm_predictions.csv'

OUTPUT_VALIDATION_CSV = os.path.join(RESULTS_FOLDER, '04_supervision.csv')          # CSV COMPLETO
OUTPUT_VALIDATION_DIF_CSV = os.path.join(RESULTS_FOLDER, '04_supervision_dif.csv')    # CSV DIFERENTES DE "Normal"
OUTPUT_HISTORICAL_UPDATE_CSV = os.path.join(RESULTS_FOLDER, '04_supervision_depurated.csv')  # CSV REGISTROS CORRECTOS

# PARÁMETROS
THRESHOLD = 0.05
MSE_ALERT_THRESHOLD = 0.1

# FLAGS
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

# 🔹 VALIDACIÓN
def validate(row):
    diff = abs(row['value'] - row['prediction'])
    if row['anomaly'] == 1:
        if diff >= THRESHOLD:
            return 'Confirmed'
        else:
            return 'Correct'
    else:
        return 'Normal'

# CREAR DATAFRAME UNIFICADO
df_val = pd.DataFrame()
df_val['value'] = df_if['agua_map07020001'] if 'agua_map07020001' in df_if.columns else df_if.iloc[:,1]
df_val['anomaly'] = df_if['anomaly']
df_val['prediction'] = df_lstm['prediction']

# APLICAR VALIDACIÓN
df_val['validation'] = df_val.apply(validate, axis=1)

# MARCAR REGISTROS CORRECTOS
df_val['infrastructure_correct'] = df_val['validation'].apply(lambda x: 1 if x=='Correct' or x=='Normal' else 0)

# 🔹 CÁLCULO DE MSE PARA REGISTROS CORRECTOS
df_correct = df_val[df_val['infrastructure_correct']==1].copy()
mse = np.mean((df_correct['value'] - df_correct['prediction'])**2)
alert_flag = 'YES' if mse > MSE_ALERT_THRESHOLD else 'NO'

if SHOW_INFO:
    print(f"[ INFO ] MSE ENTRE VALORES CORRECTOS Y PREDICCIONES: {mse:.4f}")
    if alert_flag == 'YES':
        print(f"[ ALERTA ] MSE SUPERIOR AL UMBRAL ({MSE_ALERT_THRESHOLD}) - REENTRENAMIENTO RECOMENDADO")
    else:
        print(f"[ INFO ] MSE DENTRO DE UMBRAL, PRECISIÓN ACEPTABLE")

# 🔹 AÑADIR COLUMNAS MSE Y ALERTA AL DATAFRAME COMPLETO
df_val['mse'] = mse
df_val['alert'] = alert_flag

# FILTRAR DIFERENCIAS (NO "Normal")
df_val_dif = df_val[df_val['validation'] != 'Normal'].copy()

# GUARDAR RESULTADOS
df_val.to_csv(OUTPUT_VALIDATION_CSV, index=False)
df_val_dif.to_csv(OUTPUT_VALIDATION_DIF_CSV, index=False)
df_correct.to_csv(OUTPUT_HISTORICAL_UPDATE_CSV, index=False)

if SHOW_INFO:
    print(f"[ GUARDADO ] VALIDACIÓN COMPLETA EN '{OUTPUT_VALIDATION_CSV}'")
    print(f"[ GUARDADO ] REGISTROS DIFERENTES DE 'Normal' EN '{OUTPUT_VALIDATION_DIF_CSV}'")
    print(f"[ GUARDADO ] REGISTROS CORRECTOS PARA HISTORIAL EN '{OUTPUT_HISTORICAL_UPDATE_CSV}'")
