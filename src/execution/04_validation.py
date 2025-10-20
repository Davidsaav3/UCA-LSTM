import pandas as pd           # PARA MANEJO DE DATAFRAMES
import numpy as np            # PARA MANIPULACIÓN NUMÉRICA
import os

# VARIABLES PRINCIPALES
RESULTS_FOLDER = '../../results'                                    # CARPETA PRINCIPAL DE RESULTADOS
EXECUTION_FOLDER = os.path.join(RESULTS_FOLDER, 'execution')        

INPUT_IF_CSV = os.path.join(EXECUTION_FOLDER, '01_if.csv')          # CSV CON ANOMALÍAS DE ISOLATION FOREST
INPUT_LSTM_CSV = os.path.join(EXECUTION_FOLDER, '02_lstm_predictions.csv')  # CSV CON PREDICCIONES LSTM
OUTPUT_VALIDATION_CSV = os.path.join(EXECUTION_FOLDER, '04_validation.csv') # CSV CON RESULTADO DE VALIDACIÓN
OUTPUT_ALERTS_CSV = os.path.join(EXECUTION_FOLDER, '04_alerts.csv')         # CSV CON ALERTAS DE MSE
OUTPUT_HISTORICAL_UPDATE_CSV = os.path.join(EXECUTION_FOLDER, '04_depurate_registers.csv')  # CSV REGISTROS CORRECTOS

# PARÁMETROS DE SUPERVISIÓN
THRESHOLD = 0.05               # UMBRAL PARA CONSIDERAR DIFERENCIA SIGNIFICATIVA
MSE_ALERT_THRESHOLD = 0.1       # UMBRAL PARA ALERTA AUTOMÁTICA DE MSE

# FLAGS DE CONTROL
SHOW_INFO = True                # MOSTRAR MENSAJES INFORMATIVOS

# CARGAR DATOS
df_if = pd.read_csv(INPUT_IF_CSV)        # LEER CSV DE ANOMALÍAS IF
df_lstm = pd.read_csv(INPUT_LSTM_CSV)    # LEER CSV DE PREDICCIONES LSTM

if SHOW_INFO:
    print(f"[ INFO ] DATOS IF CARGADOS: {df_if.shape}")           # INFO DE DIMENSIONES IF
    print(f"[ INFO ] DATOS LSTM CARGADOS: {df_lstm.shape}")       # INFO DE DIMENSIONES LSTM

# ASEGURAR MISMA LONGITUD ENTRE DATAFRAMES
min_len = min(len(df_if), len(df_lstm))
df_if = df_if.iloc[:min_len].copy()         # RECORTAR IF SI ES NECESARIO
df_lstm = df_lstm.iloc[:min_len].copy()     # RECORTAR LSTM SI ES NECESARIO

# RENOMBRAR COLUMNAS SI ES NECESARIO
if 'prediction' not in df_lstm.columns:
    df_lstm.rename(columns={df_lstm.columns[0]: 'prediction'}, inplace=True)  # RENOMBRAR PREDICCIÓN
if 'anomaly' not in df_if.columns:
    df_if.rename(columns={df_if.columns[0]: 'anomaly'}, inplace=True)        # RENOMBRAR ANOMALÍA

# 🔹 VALIDACIÓN: COMPARAR ANOMALÍAS Y PREDICCIONES
def validate(row):
    diff = abs(row['value'] - row['prediction'])  # DIFERENCIA ENTRE VALOR REAL Y PREDICCIÓN
    if row['anomaly'] == 1:
        if diff >= THRESHOLD:
            return 'Confirmed'      # ANOMALÍA CONFIRMADA
        else:
            return 'Correct'        # FALSO POSITIVO, REGISTRO NORMAL
    else:
        return 'Normal'            # NO ANOMALÍA

# CREAR DATAFRAME UNIFICADO PARA VALIDACIÓN
df_val = pd.DataFrame()
df_val['value'] = df_if['value'] if 'value' in df_if.columns else df_if.iloc[:,1]  # VALOR REAL
df_val['anomaly'] = df_if['anomaly']                                              # INDICADOR IF
df_val['prediction'] = df_lstm['prediction']                                       # PREDICCIÓN LSTM

# APLICAR VALIDACIÓN
df_val['validation'] = df_val.apply(validate, axis=1)  # APLICAR FUNCION VALIDATE

# CREAR COLUMNA "Infrastructure Correct"
df_val['infrastructure_correct'] = df_val['validation'].apply(lambda x: 1 if x=='Correct' or x=='Normal' else 0)  # MARCAR REGISTROS CORRECTOS

# 🔹 CÁLCULO DE MSE PARA REGISTROS CORRECTOS
df_correct = df_val[df_val['infrastructure_correct']==1].copy()    # FILTRAR REGISTROS CORRECTOS
mse = np.mean((df_correct['value'] - df_correct['prediction'])**2)  # CALCULAR MSE
if SHOW_INFO:
    print(f"[ INFO ] MSE ENTRE VALORES CORRECTOS Y PREDICCIONES: {mse:.4f}")  # MOSTRAR MSE

# 🔹 GENERAR ALERTAS AUTOMÁTICAS SI MSE > UMBRAL
df_alerts = pd.DataFrame()
df_alerts['mse'] = [mse]  
df_alerts['alert'] = ['YES' if mse > MSE_ALERT_THRESHOLD else 'NO']  # INDICAR ALERTA
if SHOW_INFO:
    if mse > MSE_ALERT_THRESHOLD:
        print(f"[ ALERTA ] MSE SUPERIOR AL UMBRAL ({MSE_ALERT_THRESHOLD}) - REENTRENAMIENTO RECOMENDADO")
    else:
        print(f"[ INFO ] MSE DENTRO DE UMBRAL, PRECISIÓN ACEPTABLE")

# 🔹 ACTUALIZACIÓN DEL REPOSITORIO HISTÓRICO CON REGISTROS CORRECTOS
df_historical_update = df_correct.copy()  # COPIAR REGISTROS DEPURADOS PARA HISTORIAL

# GUARDAR RESULTADOS
df_val.to_csv(OUTPUT_VALIDATION_CSV, index=False)               # GUARDAR VALIDACIÓN
df_alerts.to_csv(OUTPUT_ALERTS_CSV, index=False)                # GUARDAR ALERTAS
df_historical_update.to_csv(OUTPUT_HISTORICAL_UPDATE_CSV, index=False)  # GUARDAR REGISTROS CORRECTOS

if SHOW_INFO:
    print(f"[ GUARDADO ] VALIDACIÓN COMPLETA EN '{OUTPUT_VALIDATION_CSV}'")
    print(f"[ GUARDADO ] ALERTAS DE MSE EN '{OUTPUT_ALERTS_CSV}'")
    print(f"[ GUARDADO ] REGISTROS CORRECTOS PARA HISTORIAL EN '{OUTPUT_HISTORICAL_UPDATE_CSV}'")