import pandas as pd  # PARA MANEJO DE DATAFRAMES 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler  # ESCALADORES DE CARACTERÍSTICAS

# PARÁMETROS 
INPUT_FILE = '../../results/preparation/03_codification.csv'  # CSV DE ENTRADA YA CODIFICADO
OUTPUT_FILE = '../../results/preparation/04_scale.csv'        # CSV ESCALADO QUE SE GUARDARÁ
SCALER_TYPE = 'standard'                                      # TIPO DE ESCALADO: 'standard', 'minmax', 'robust', 'maxabs'
SHOW_INFO = True                                              # MOSTRAR MENSAJES DE INFO EN PANTALLA
SAVE_INTERMEDIATE = True                                     # GUARDAR DATASET INTERMEDIO ANTES DE ESCALAR
FEATURES_TO_SCALE = None                                      # COLUMNAS A ESCALAR, NONE = TODAS
CLIP_VALUES = False                                           # RECORTAR VALORES EXTREMOS DESPUÉS DEL ESCALADO
CLIP_MIN = 0                                                  # VALOR MÍNIMO SI SE RECORTAN EXTREMOS
CLIP_MAX = 1                                                  # VALOR MÁXIMO SI SE RECORTAN EXTREMOS

# 🔹 MODIFICADO PARA LSTM → Se fuerza MinMaxScaler como escalador por defecto
SCALER_TYPE = 'minmax'  # 🔹 MODIFICADO PARA LSTM

# CARGAR DATASET
df = pd.read_csv(INPUT_FILE, low_memory=False)                                   # LEER CSV DE ENTRADA
if SHOW_INFO:
    print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# 🔹 MODIFICADO PARA LSTM → Asegurar que está ordenado temporalmente
if 'datetime' in df.columns:
    df = df.sort_values('datetime')  # 🔹 MODIFICADO PARA LSTM
    if SHOW_INFO:
        print("[ INFO ] DATASET ORDENADO POR COLUMNA 'datetime'")  # 🔹 MODIFICADO PARA LSTM

# SELECCIONAR COLUMNAS A ESCALAR
if FEATURES_TO_SCALE is None:
    FEATURES_TO_SCALE = df.columns.tolist()  # SI NO SE INDICAN COLUMNAS, SE ESCALAN TODAS
if SHOW_INFO:
    print(f"[ INFO ] COLUMNAS A ESCALAR: {len(FEATURES_TO_SCALE)}")

# 🔹 MODIFICADO PARA LSTM → Evitar escalar columnas temporales (datetime)
FEATURES_TO_SCALE = [c for c in FEATURES_TO_SCALE if c != 'datetime']  # 🔹 MODIFICADO PARA LSTM

# SELECCIÓN DEL ESCALADOR SEGÚN TIPO
if SCALER_TYPE == 'standard':
    scaler = StandardScaler()  
    # ESCALA CADA COLUMNA RESTANDO SU MEDIA Y DIVIDIENDO ENTRE DESVIACIÓN ESTÁNDAR
    # ADECUADO PARA DATOS CON DISTRIBUCIÓN NORMAL Y ALGORITMOS SENSIBLES A ESCALA
elif SCALER_TYPE == 'minmax':
    scaler = MinMaxScaler()    
    # NORMALIZA ENTRE 0 Y 1, ÚTIL PARA REDES NEURONALES Y MÉTODOS QUE REQUIEREN RANGO FIJO
elif SCALER_TYPE == 'robust':
    scaler = RobustScaler()    
    # ESCALA UTILIZANDO MEDIANA Y RANGO INTERCUARTÍLICO
    # MENOS SENSIBLE A VALORES EXTREMOS O OUTLIERS
elif SCALER_TYPE == 'maxabs':
    scaler = MaxAbsScaler()    
    # ESCALA ENTRE -1 Y 1, MANTENIENDO SIGNO, ÚTIL PARA DATOS QUE PUEDEN SER NEGATIVOS
else:
    raise ValueError(f"[ ERROR ] ESCALADOR DESCONOCIDO: {SCALER_TYPE}")

# GUARDAR DATASET INTERMEDIO (OPCIONAL)
if SAVE_INTERMEDIATE:
    intermediate_file = OUTPUT_FILE.replace('.csv','_intermediate.csv')
    df.to_csv(intermediate_file, index=False)
    if SHOW_INFO:
        print(f"[ GUARDADO ] DATASET INTERMEDIO EN '{intermediate_file}'")

# 🔹 MODIFICADO PARA LSTM → Escalar solo características numéricas
numeric_cols = df[FEATURES_TO_SCALE].select_dtypes(include=['number']).columns.tolist()  # 🔹 MODIFICADO PARA LSTM
if SHOW_INFO:
    print(f"[ INFO ] COLUMNAS NUMÉRICAS A ESCALAR: {len(numeric_cols)}")  # 🔹 MODIFICADO PARA LSTM

# 🔧 MODIFICACIÓN: Evitar columnas con todos NaN antes de escalar
numeric_cols_clean = []
for col in numeric_cols:
    if df[col].isna().all():
        if SHOW_INFO:
            print(f"[ AVISO ] Columna '{col}' contiene solo NaN y no se escalará")  # INFORMACIÓN
        continue
    numeric_cols_clean.append(col)

# CREAR COPIA DEL DATASET PARA ESCALAR
df_scaled = df.copy()  # COPIA PARA NO MODIFICAR DATASET ORIGINAL
df_scaled[numeric_cols_clean] = scaler.fit_transform(df_scaled[numeric_cols_clean]) # 🔹 MODIFICADO PARA LSTM → Escala solo numéricas
if SHOW_INFO:
    print(f"[ INFO ] DATASET ESCALADO USANDO '{SCALER_TYPE}'")

# RECORTAR VALORES EXTREMOS (OPCIONAL)
if CLIP_VALUES:
    df_scaled[FEATURES_TO_SCALE] = df_scaled[FEATURES_TO_SCALE].clip(CLIP_MIN, CLIP_MAX) # MAX Y MIN
    if SHOW_INFO:
        print(f"[ INFO ] VALORES RECORTADOS ENTRE {CLIP_MIN} Y {CLIP_MAX}")

# GUARDAR DATASET ESCALADO FINAL
df_scaled.to_csv(OUTPUT_FILE, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] DATASET ESCALADO EN '{OUTPUT_FILE}'")
