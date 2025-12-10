import pandas as pd 
import numpy as np   
import os          
import sys
sys.stdout.reconfigure(encoding='utf-8')

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results/03_execution/04_supervision'  # CARPETA DE RESULTADOS
INPUT_IF_CSV = '../../results/03_execution/01_classification/01_if.csv'  # CSV IF
INPUT_LSTM_CSV = '../../results/03_execution/02_prediction/02_lstm_predictions.csv'  # CSV LSTM

OUTPUT_VALIDATION_CSV = os.path.join(RESULTS_FOLDER, '04_supervision.csv')          # CSV COMPLETO
OUTPUT_VALIDATION_DIF_CSV = os.path.join(RESULTS_FOLDER, '04_supervision_dif.csv') # CSV DIFERENCIAS NO "Normal"
OUTPUT_HISTORICAL_UPDATE_CSV = os.path.join(RESULTS_FOLDER, '04_supervision_depurated.csv')  # CSV REGISTROS CORRECTOS

# PARÁMETROS
THRESHOLD = 0.05               # UMBRAL PARA VALIDACIÓN DE DIFERENCIA
MSE_ALERT_THRESHOLD = 0.1      # UMBRAL PARA ALERTA DE MSE

# FLAGS
SHOW_INFO = True                # TRUE = IMPRIMIR INFO EN PANTALLA

# CARGAR DATOS
df_if = pd.read_csv(INPUT_IF_CSV)    # LEER CSV IF
df_lstm = pd.read_csv(INPUT_LSTM_CSV)  # LEER CSV PREDICCIONES LSTM

if SHOW_INFO:
    print(f"[ INFO ] DATOS IF CARGADOS: {df_if.shape}")       # MOSTRAR DIMENSIONES IF
    print(f"[ INFO ] DATOS LSTM CARGADOS: {df_lstm.shape}")   # MOSTRAR DIMENSIONES LSTM

# ASEGURAR MISMA LONGITUD ENTRE DATAFRAMES
min_len = min(len(df_if), len(df_lstm))  
df_if = df_if.iloc[:min_len].copy()
df_lstm = df_lstm.iloc[:min_len].copy()

# RENOMBRAR COLUMNAS SI ES NECESARIO
if 'pred_step10' not in df_lstm.columns:
    df_lstm.rename(columns={df_lstm.columns[0]: 'pred_step10'}, inplace=True)  # RENOMBRAR COLUMNA PREDICCIÓN
if 'anomaly' not in df_if.columns:
    df_if.rename(columns={df_if.columns[0]: 'anomaly'}, inplace=True)          # RENOMBRAR COLUMNA ANOMALÍA

# 🔹 FUNCION DE VALIDACIÓN
def validate(row):
    diff = abs(row['value'] - row['pred_step10'])  # CALCULAR DIFERENCIA ABSOLUTA
    if row['anomaly'] == 1:
        if diff >= THRESHOLD:
            return 'Confirmed'   # ANOMALÍA CONFIRMADA
        else:
            return 'Correct'     # ANOMALÍA CORRECTA
    else:
        return 'Normal'          # REGISTRO NORMAL

# CREAR DATAFRAME UNIFICADO
df_val = pd.DataFrame()
df_val['value'] = df_if['wifi_inal_sf_1_39'] if 'wifi_inal_sf_1_39' in df_if.columns else df_if.iloc[:,1]  # VALORES
df_val['anomaly'] = df_if['anomaly']       # ANOMALÍAS
df_val['pred_step10'] = df_lstm['pred_step10']  # PREDICCIONES

# APLICAR VALIDACIÓN
df_val['validation'] = df_val.apply(validate, axis=1)  # VALIDAR CADA FILA

# MARCAR REGISTROS CORRECTOS
df_val['infrastructure_correct'] = df_val['validation'].apply(lambda x: 1 if x=='Correct' or x=='Normal' else 0)  # FLAG CORRECTOS

# 🔹 CÁLCULO DE MSE PARA REGISTROS CORRECTOS
df_correct = df_val[df_val['infrastructure_correct']==1].copy()  # FILTRAR CORRECTOS
mse = np.mean((df_correct['value'] - df_correct['pred_step10'])**2)  # CALCULAR MSE
alert_flag = 'YES' if mse > MSE_ALERT_THRESHOLD else 'NO'  # FLAG ALERTA

if SHOW_INFO:
    print(f"[ INFO ] MSE ENTRE VALORES CORRECTOS Y PREDICCIONES: {mse:.4f}")  # MOSTRAR MSE
    if alert_flag == 'YES':
        print(f"[ ALERTA ] MSE SUPERIOR AL UMBRAL ({MSE_ALERT_THRESHOLD}) - REENTRENAMIENTO RECOMENDADO")
    else:
        print(f"[ INFO ] MSE DENTRO DE UMBRAL, PRECISIÓN ACEPTABLE")

# 🔹 AÑADIR COLUMNAS MSE Y ALERTA AL DATAFRAME COMPLETO
df_val['mse'] = mse             # GUARDAR MSE
df_val['alert'] = alert_flag    # GUARDAR ALERTA

# FILTRAR DIFERENCIAS (NO "Normal")
df_val_dif = df_val[df_val['validation'] != 'Normal'].copy()  # FILTRAR REGISTROS DIFERENTES

# GUARDAR RESULTADOS
df_val.to_csv(OUTPUT_VALIDATION_CSV, index=False)           # CSV COMPLETO
df_val_dif.to_csv(OUTPUT_VALIDATION_DIF_CSV, index=False)   # CSV DIFERENCIAS
df_correct.to_csv(OUTPUT_HISTORICAL_UPDATE_CSV, index=False)  # CSV CORRECTOS

if SHOW_INFO:
    print(f"[ GUARDADO ] VALIDACIÓN COMPLETA EN '{OUTPUT_VALIDATION_CSV}'")
    print(f"[ GUARDADO ] REGISTROS DIFERENTES DE 'Normal' EN '{OUTPUT_VALIDATION_DIF_CSV}'")
    print(f"[ GUARDADO ] REGISTROS CORRECTOS PARA HISTORIAL EN '{OUTPUT_HISTORICAL_UPDATE_CSV}'")
