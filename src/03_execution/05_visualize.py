import pandas as pd                  # IMPORTA PANDAS PARA MANEJO DE DATOS
import matplotlib.pyplot as plt      # IMPORTA MATPLOTLIB PARA CREAR GRÁFICAS
import seaborn as sns                # IMPORTA SEABORN PARA ESTILO Y VISUALIZACIÓN AVANZADA
import os                            # IMPORTA OS PARA GESTIONAR RUTAS Y CARPETAS
import glob                          # IMPORTA GLOB PARA BUSCAR ARCHIVOS CON PATRONES
import numpy as np

# DEFINE LAS RUTAS Y PARÁMETROS GENERALES DEL SCRIPT
01_if_CSV = '../../results/03_execution/01_if.csv'        # ARCHIVO GLOBAL CON RESULTADOS DEL ISOLATION FOREST
IF_01_CSV = '../../results/03_execution/01_if_anomaly.csv'                # ARCHIVO CON SECUENCIAS DE ANOMALÍAS DETECTADAS
RESULTS_SUMMARY_CSV = '../../results/03_execution/05_results.csv' # ARCHIVO CON LAS MÉTRICAS DE RENDIMIENTO
RESULTS_FOLDER = '../../results/03_execution/plots'               # CARPETA DONDE SE GUARDARÁN LOS GRÁFICOS

FEATURE_TO_PLOT = 'nivel_plaxiquet'                           # VARIABLE PRINCIPAL A VISUALIZAR EN LOS GRÁFICOS
SAVE_FIGURES = True                                            # DEFINE SI SE GUARDAN LAS FIGURAS EN ARCHIVOS
SHOW_FIGURES = False                                           # DEFINE SI SE MUESTRAN EN PANTALLA LAS FIGURAS
STYLE = 'whitegrid'                                            # ESTILO VISUAL PARA SEABORN

# CREA LA CARPETA DE RESULTADOS SI NO EXISTE
os.makedirs(RESULTS_FOLDER, exist_ok=True)                     # CREA LA CARPETA SI NO EXISTE
print(f"[ INFO ] CARPETA '{RESULTS_FOLDER}' CREADA SI NO EXISTÍA")  # IMPRIME CONFIRMACIÓN EN CONSOLA

# CONFIGURA EL ESTILO Y EL TAMAÑO DE LAS FIGURAS
sns.set_style(STYLE)                                           # APLICA ESTILO VISUAL A SEABORN
plt.rcParams['figure.figsize'] = (12, 6)                       # DEFINE TAMAÑO POR DEFECTO DE LAS FIGURAS

# CARGA EL ARCHIVO PRINCIPAL CON LOS RESULTADOS DEL MODELO GLOBAL
df_if = pd.read_csv(01_if_CSV)                             # LEE EL ARCHIVO CSV PRINCIPAL

# CONVIERTE VARIABLES A TIPO ENTERO PARA ASEGURAR CONSISTENCIA
df_if['anomaly'] = df_if['anomaly'].astype(int)                # CONVIERTE LA COLUMNA 'anomaly' A ENTERO
df_if['is_anomaly'] = df_if['is_anomaly'].astype(int)          # CONVIERTE LA COLUMNA 'is_anomaly' A ENTERO                                   # CREA COLUMNA 'cluster' CON VALOR 0

# 1. GRÁFICO: ANOMALÍAS DETECTADAS VS REALES
plt.figure(figsize=(18, 6))                                    # CREA UNA NUEVA FIGURA DE TAMAÑO GRANDE
sns.scatterplot(                                               
    data=df_if,                                                 # USA EL DATAFRAME GLOBAL
    x='datetime',                                               # EJE X: FECHA Y HORA
    y=FEATURE_TO_PLOT,                                          # EJE Y: VARIABLE PRINCIPAL
    hue='anomaly',                                              # COLOR SEGÚN SI ES ANOMALÍA O NO
    palette={0: 'blue', 1: 'red'},                              # DEFINE COLORES (NORMAL=GRIS, ANOMALÍA=ROJO)
    alpha=0.7                                                   # DEFINE TRANSPARENCIA
)
plt.title(f"Anomalies Detected vs Real: {FEATURE_TO_PLOT.upper()}")  # TÍTULO DEL GRÁFICO
plt.xlabel("Datetime")                                          # ETIQUETA DEL EJE X
plt.ylabel(FEATURE_TO_PLOT.upper())                             # ETIQUETA DEL EJE Y
plt.xticks(rotation=45)                                         # ROTA ETIQUETAS DE TIEMPO
plt.legend(title='Anomaly', labels=['Real', 'Detected'])      # DEFINE LEYENDA
if SAVE_FIGURES:                                                # SI SE DEBE GUARDAR LA FIGURA
    plt.savefig(f"{RESULTS_FOLDER}/01_anomalies_vs_real.png", dpi=300, bbox_inches='tight')  # GUARDA EL GRÁFICO
plt.close()                                                     # CIERRA LA FIGURA PARA LIBERAR MEMORIA

# 2. HEATMAP DE CORRELACIÓN ENTRE VARIABLES NUMÉRICAS
numeric_cols = df_if.select_dtypes(include=['float64', 'int64']).columns  # SELECCIONA SOLO COLUMNAS NUMÉRICAS
plt.figure(figsize=(30, 18))                                  # CREA UNA FIGURA DE GRAN TAMAÑO
sns.heatmap(                                                  # CREA UN HEATMAP DE CORRELACIONES
    df_if[numeric_cols].corr(),                               # CALCULA MATRIZ DE CORRELACIÓN
    annot=True, fmt=".2f", cmap='coolwarm',                   # MUESTRA VALORES NUMÉRICOS CON FORMATO
    cbar=True, annot_kws={"size": 3}, linewidths=0.3, linecolor='white'  # CONFIGURA DETALLES VISUALES
)
plt.title("Correlation Matrix - All Numeric Features", fontsize=14)  # TÍTULO DEL HEATMAP
plt.xticks(rotation=45, ha='right', fontsize=4)               # ROTA Y AJUSTA ETIQUETAS X
plt.yticks(rotation=0, fontsize=4)                            # AJUSTA ETIQUETAS Y
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/02_correlation_matrix.png", dpi=300, bbox_inches='tight')  # GUARDA EL HEATMAP
plt.close()

# 3. HISTOGRAMA DE SCORES DE ANOMALÍAS
df_if = pd.read_csv(IF_01_CSV)                             # LEE EL ARCHIVO CSV PRINCIPAL

plt.figure()                                                 # CREA NUEVA FIGURA
sns.histplot(df_if['anomaly_score'], bins=50, kde=True, color='red')  # CREA HISTOGRAMA CON CURVA KDE
plt.title("Distribution of Anomaly Scores")                  # TÍTULO DEL GRÁFICO
plt.xlabel("Anomaly Score")                                  # ETIQUETA EJE X
plt.ylabel("Frequency")                                      # ETIQUETA EJE Y
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/03_anomaly_score_distribution.png", dpi=300)  # GUARDA EL HISTOGRAMA
plt.close()

# CARGA MÉTRICAS DE RENDIMIENTO
df_summary = pd.read_csv(RESULTS_SUMMARY_CSV)          # Carga el CSV con las métricas calculadas de los modelos
df_summary.set_index('file', inplace=True)             # Usa el nombre del archivo (método o modelo) como índice

# 7. GRÁFICO DE MÉTRICAS DE RENDIMIENTO
metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'mcc']  # Lista de métricas a visualizar
df_summary[metrics].plot(kind='bar', figsize=(16, 6))              # Gráfico de barras comparando el rendimiento por método
plt.title("Performance Metrics per Method / File")                 # Título del gráfico
plt.ylabel("Score")                                                # Etiqueta eje Y
plt.ylim(0, 1.1)                                                   # Escala del eje Y de 0 a 1.1 para ver bien las diferencias
plt.xticks(rotation=45, ha='right')                                # Rota las etiquetas X
plt.legend(title='Metric')                                         # Muestra la leyenda de métricas
plt.tight_layout()                                                 # Ajusta márgenes
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/07_metrics_comparison.png", dpi=300)  # Guarda la figura
plt.close()

# 8. RATIO DE DETECCIÓN VS FALSOS POSITIVOS
ratio_metrics = ['anomalies_real','anomalies_detected','detections_correct', 'total_coincidences']                    # Selecciona las métricas de ratio
df_summary[ratio_metrics].plot(kind='bar',figsize=(16, 6),)
plt.title("Ratio Detection vs False Positives")                    # Título del gráfico
plt.ylabel("Ratio")                                                # Etiqueta eje Y
plt.xticks(rotation=45, ha='right')                                # Rota etiquetas X
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/08_ratio_detection_fp.png", dpi=300)  # Guarda la figura
plt.close()

# 9. TRUE POSITIVES, FALSE POSITIVES, FALSE NEGATIVES
df_summary[['detections_correct', 'false_positives', 'false_negatives']].plot(
    kind='bar',
    figsize=(16, 6),
    color=['blue', 'green', 'red']  # Verde = correct detections, Rojo = FP, Naranja = FN
)
plt.title("True Positives, False Positives and False Negatives")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f"{RESULTS_FOLDER}/09_tp_fp_fn.png", dpi=300)
plt.close()

# MENSAJE FINAL DE CONFIRMACIÓN
print("Visualizations saved in:", RESULTS_FOLDER)  # Mensaje en consola confirmando la ubicación de las gráficas
