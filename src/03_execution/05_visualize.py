import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding='utf-8')

#  PARÁMETROS DEL SCRIPT 
IF_CSV = '../../results/03_execution/01_classification/01_if.csv'
LSTM_VALUES_CSV = '../../results/03_execution/02_prediction/02_lstm_predictions.csv'      # CSV CON VALORES Y PREDICCIONES
LSTM_HISTORY_CSV = '../../results/03_execution/02_prediction/02_lstm_history.csv'        # CSV CON LOSS Y MAE
DIAGNOSTIC_CSV = '../../results/03_execution/03_diagnostic/03_diagnostic.csv'       # CSV CON VALUE, ANOMALY, PREDICTION, DIAGNOSTIC, DIFF
ALERT_CSV = '../../results/03_execution/04_supervision/04_supervision.csv'                 # CSV CON VALUE, ANOMALY, PREDICTION, VALIDATION, INF_CORRECT, MSE, ALERT
RESULTS_FOLDER = '../../results/03_execution/plots'                                      # CARPETA DONDE SE GUARDARÁN LOS GRÁFICOS
SAVE_FIGURE = True                                                                       # GUARDAR LAS FIGURAS EN DISCO
SHOW_FIGURE = False                                                                      # NO MOSTRAR FIGURA EN PANTALLA
FEATURE_NAME = 'wifi_inal_sf_1_39'                                                        # NOMBRE DE LA VARIABLE
STYLE = 'whitegrid'                                                                      # ESTILO DE SEABORN

#  CREAR CARPETA DE RESULTADOS 
os.makedirs(RESULTS_FOLDER, exist_ok=True)
print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTÍA")

#  CONFIGURAR ESTILO 
sns.set_style(STYLE)
plt.rcParams['figure.figsize'] = (20, 6)

# ================== GRÁFICO 1: ANOMALY SCORE Y VALOR REAL DEL IF ==================
df_if_plot = pd.read_csv(IF_CSV)
print(f"[ INFO ] CSV '{IF_CSV}' CARGADO CON {df_if_plot.shape[0]} FILAS Y {df_if_plot.shape[1]} COLUMNAS")

# Columnas a graficar: anomaly_score y valor real
columns_to_plot = ['anomaly_score', 'wifi_inal_sf_1_39']

for col in columns_to_plot:
    plt.figure(figsize=(20,6))
    plt.plot(df_if_plot[col], label=col.replace('_', ' ').title(), color='blue', linewidth=2)

    # Puntos rojos donde hay anomalía
    anomaly_mask = df_if_plot['anomaly'] == 1
    plt.scatter(df_if_plot.index[anomaly_mask], df_if_plot[col][anomaly_mask],
                color='red', marker='o', s=50, label='Anomalía Detectada', zorder=5)

    plt.title(f"{col.replace('_', ' ').title()} vs Registro (Isolation Forest)")
    plt.xlabel("Índice de Registro")
    plt.ylabel(col.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(RESULTS_FOLDER, f"01_classification_{col}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[ INFO ] FIGURA GUARDADA EN '{output_path}'")
    plt.close()

#  GRÁFICO 2: VALOR VS PREDICCIÓN LSTM -------------------------------------------------------
df_values = pd.read_csv(LSTM_VALUES_CSV)
print(f"[ INFO ] CSV '{LSTM_VALUES_CSV}' CARGADO CON {df_values.shape[0]} FILAS Y {df_values.shape[1]} COLUMNAS")

# Soporte para multi-step: usar pred_step1 si existe, sino 'prediction'
if 'pred_step1' in df_values.columns:
    df_values['prediction'] = df_values['pred_step1']
    print("[ INFO ] Detectado formato multi-step: usando 'pred_step1' como 'prediction'")
elif 'prediction' in df_values.columns:
    df_values['prediction'] = df_values['prediction']
else:
    raise KeyError("No se encontró columna 'prediction' ni 'pred_step1' en el CSV de predicciones")

# Convertir a numérico y manejar NaN
df_values['prediction'] = pd.to_numeric(df_values['prediction'], errors='coerce')

plt.figure(figsize=(20,6))
plt.plot(df_values[FEATURE_NAME], label='Valor Real', color='blue', linewidth=2)
plt.plot(df_values['prediction'], label='Predicción LSTM', color='red', linewidth=2, linestyle='--')
plt.title(f"Valor vs Predicción LSTM: {FEATURE_NAME}")
plt.xlabel("Índice de Registro")
plt.ylabel(FEATURE_NAME)
plt.legend()
plt.grid(True)

if SAVE_FIGURE:
    output_path_values = os.path.join(RESULTS_FOLDER, f"02_prediction_{FEATURE_NAME}.png")
    plt.savefig(output_path_values, dpi=300, bbox_inches='tight')
    print(f"[ INFO ] FIGURA GUARDADA EN '{output_path_values}'")
plt.close()

#  GRÁFICO 2.1: LOSS Y MAE -----------------------------------------------------------------------
df_history = pd.read_csv(LSTM_HISTORY_CSV)
print(f"[ INFO ] CSV '{LSTM_HISTORY_CSV}' CARGADO CON {df_history.shape[0]} FILAS Y {df_history.shape[1]} COLUMNAS")

plt.figure(figsize=(20,6))
plt.plot(df_history['loss'], label='Loss', color='blue', linewidth=2)
plt.plot(df_history['mae'], label='MAE', color='red', linewidth=2, linestyle='--')
plt.title("Evolución de Loss y MAE durante entrenamiento LSTM")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)

if SAVE_FIGURE:
    output_path_history = os.path.join(RESULTS_FOLDER, "02_prediction_loss_mae.png")
    plt.savefig(output_path_history, dpi=300, bbox_inches='tight')
    print(f"[ INFO ] FIGURA GUARDADA EN '{output_path_history}'")
plt.close()

#  GRÁFICO 3: VALUE, PREDICTION, DIFF CON ANOMALÍAS Y DIAGNÓSTICO -----------------------------------------
df_diag = pd.read_csv(DIAGNOSTIC_CSV)
print(f"[ INFO ] CSV '{DIAGNOSTIC_CSV}' CARGADO CON {df_diag.shape[0]} FILAS Y {df_diag.shape[1]} COLUMNAS")

# Soporte multi-step en diagnostic.csv (asumiendo que también tiene pred_step1)
if 'pred_step1' in df_diag.columns:
    df_diag['prediction'] = df_diag['pred_step1']
    print("[ INFO ] Usando 'pred_step1' como 'prediction' en diagnostic.csv")
elif 'prediction' in df_diag.columns:
    df_diag['prediction'] = df_diag['prediction']
else:
    df_diag['prediction'] = np.nan  # fallback si no existe

df_diag['prediction'] = pd.to_numeric(df_diag['prediction'], errors='coerce')
df_diag['diff'] = pd.to_numeric(df_diag.get('diff', np.nan), errors='coerce')

plt,plt.figure(figsize=(50,10))

# Líneas principales
plt.plot(df_diag['value'], label='Valor Real', color='blue', linewidth=2)
plt.plot(df_diag['prediction'], label='Predicción LSTM', color='red', linewidth=2, linestyle='--')
plt.plot(df_diag['diff'], label='Diferencia (diff)', color='green', linewidth=2, linestyle=':')

# Marcadores según anomaly
anomaly_mask = df_diag['anomaly'] == 1
plt.scatter(df_diag.index[anomaly_mask], df_diag['value'][anomaly_mask],
            color='orange', marker='o', label='Anomalía Detectada', s=50, zorder=5)

# Marcadores según diagnostic
diagnostic_colors = {
    'Confirmed': 'red',
    'Falso Positivo': 'purple',
    'Falso Negativo': 'brown'
}
for diag, color in diagnostic_colors.items():
    mask = df_diag['diagnostic'] == diag
    plt.scatter(df_diag.index[mask], df_diag['value'][mask],
                color=color, marker='x', label=f'Diagnóstico: {diag}', s=40, zorder=6)

plt.title("Valor Real, Predicción LSTM, Diferencia (diff) y Diagnóstico")
plt.xlabel("Índice de Registro")
plt.ylabel("Valor")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Mueve leyenda fuera de gráfico
plt.grid(True)

if SAVE_FIGURE:
    output_path_diag = os.path.join(RESULTS_FOLDER, "03_diagnostic.png")
    plt.savefig(output_path_diag, dpi=300, bbox_inches='tight')
    print(f"[ INFO ] FIGURA GUARDADA EN '{output_path_diag}'")
plt.close()

#  GRÁFICO 4: VALUE, PREDICTION Y MSE CON ANOMALÍAS, VALIDATION, INF_CORRECT Y ALERT ---------------------------------
df_alert = pd.read_csv(ALERT_CSV)
print(f"[ INFO ] CSV '{ALERT_CSV}' CARGADO CON {df_alert.shape[0]} FILAS Y {df_alert.shape[1]} COLUMNAS")

# Soporte multi-step en alert.csv
if 'pred_step1' in df_alert.columns:
    df_alert['prediction'] = df_alert['pred_step1']
    print("[ INFO ] Usando 'pred_step1' como 'prediction' en alert.csv")
elif 'prediction' in df_alert.columns:
    df_alert['prediction'] = df_alert['prediction']
else:
    df_alert['prediction'] = np.nan

df_alert['prediction'] = pd.to_numeric(df_alert['prediction'], errors='coerce')
df_alert['mse'] = pd.to_numeric(df_alert.get('mse', np.nan), errors='coerce')

plt.figure(figsize=(50,10))

# Líneas principales
plt.plot(df_alert['value'], label='Valor Real', color='blue', linewidth=2)
plt.plot(df_alert['prediction'], label='Predicción LSTM', color='red', linewidth=2, linestyle='--')
plt.plot(df_alert['mse'], label='MSE', color='green', linewidth=2, linestyle=':')

# Marcadores según anomaly
anomaly_mask = df_alert['anomaly'] == 1
plt.scatter(df_alert.index[anomaly_mask], df_alert['value'][anomaly_mask],
            color='orange', marker='o', label='Anomalía', s=50, zorder=5)

# Marcadores según validation
validation_colors = {
    'Confirmed': 'red',
    'Correct': 'gray'
}
for val, color in validation_colors.items():
    mask = df_alert['validation'] == val
    plt.scatter(df_alert.index[mask], df_alert['value'][mask],
                color=color, marker='x', label=f'Validation: {val}', s=40, zorder=6)
    
# Solo pintar cuando infrastructure_correct == 0
mask_ic0 = df_alert['infrastructure_correct'] == 0
plt.scatter(df_alert.index[mask_ic0],
            df_alert['value'][mask_ic0],
            color='brown', marker='^', label='Infrastructure Correct: 0', s=40, zorder=7)

# Marcadores solo cuando alert == YES
mask_alert_yes = df_alert['alert'] == 'YES'
plt.scatter(df_alert.index[mask_alert_yes],
            df_alert['value'][mask_alert_yes],
            color='purple', marker='s', label='Alert: YES', s=40, zorder=8)

plt.title("Valor Real, Predicción LSTM, MSE y Categorías de Anomalías/Validación/Infraestructura/Alert")
plt.xlabel("Índice de Registro")
plt.ylabel("Valor / MSE")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Leyenda fuera del gráfico
plt.grid(True)

if SAVE_FIGURE:
    output_path_alert = os.path.join(RESULTS_FOLDER, "04_validation.png")
    plt.savefig(output_path_alert, dpi=300, bbox_inches='tight')
    print(f"[ INFO ] FIGURA GUARDADA EN '{output_path_alert}'")
plt.close()

print("[ INFO ] SCRIPT FINALIZADO - LOS 4 GRÁFICOS HAN SIDO GUARDADOS")