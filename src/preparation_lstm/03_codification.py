import pandas as pd  # PARA MANEJO DE DATAFRAMES
import json          # PARA GUARDAR INFORMACIÓN DE CAMBIOS DE COLUMNAS

# PARÁMETROS 
INPUT_FILE = '../../results/preparation/02_nulls.csv'          # ARCHIVO DE ENTRADA DEL DATASET ORIGINAL
OUTPUT_FILE = '../../results/preparation/03_codification.csv'  # ARCHIVO DE SALIDA DEL DATASET PROCESADO
AUX_FILE = '../../results/preparation/03_aux.json'             # ARCHIVO JSON PARA REGISTRAR CAMBIOS REALIZADOS

# LISTA DE COLUMNAS RELACIONADAS CON FECHA U HORA QUE NO SE CODIFICARÁN
FECHA_COLS = [
    'datetime', 'date', 'time',
    'aemet_temperatura_minima_hora'
    'aemet_temperatura_maxima_hora',
    'aemet_viento_hora',
    'aemet_presion_maxima_hora',
    'aemet_presion_minima_hora',
    'aemet_humediad_maxima_hora',
    'aemet_humediad_minima_hora'
]

ONE_HOT_DROP_FIRST = True   # ELIMINA LA PRIMERA CATEGORÍA PARA EVITAR MULTICOLINEALIDAD (Cuando dos o más variables están fuertemente correlacionadas)
REPLACE_SPACES = '-'        # REEMPLAZA ESPACIOS EN LOS NOMBRES DE NUEVAS COLUMNAS

# CARGAR DATASET DESDE EL CSV DE ENTRADA
df = pd.read_csv(INPUT_FILE, low_memory=False)  
print(f"[ INFO ] Dataset cargado: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")  # MUESTRA DIMENSIONES DEL DATASET

# 🔧 CAMBIO: ORDENAR POR COLUMNA TEMPORAL SI EXISTE (IMPORTANTE PARA LSTM)
if 'datetime' in df.columns:
    df = df.sort_values(by='datetime').reset_index(drop=True)
    print("[ INFO ] Dataset ordenado por columna temporal 'datetime'")

# INICIALIZAR ARCHIVO AUX PARA REGISTRAR MODIFICACIONES REALIZADAS
columns_changes = {
    'fecha_to_timestamp': [],     # COLUMNAS DE FECHA CONVERTIDAS A TIMESTAMP
    'hora_to_seconds': [],        # COLUMNAS DE HORA CONVERTIDAS A SEGUNDOS
    'one_hot': []                 # COLUMNAS CODIFICADAS MEDIANTE ONE-HOT
}

# 🔧 CAMBIO: NO CONVERTIR COLUMNAS DE FECHA A TIMESTAMP PARA LSTM (SE MANTIENEN COMO DATETIME)
# En modelos LSTM la secuencia temporal se preserva en formato datetime; 
# por tanto, no se transforman a segundos desde 1970 para evitar pérdida de información temporal.

# 🔧 CAMBIO: Especificar formato de fecha conocido para evitar warning
for col in ['datetime', 'date']:
    if col in df.columns:
        # INTENTA FORMATO CON DÍA/MES/AÑO Y POSIBLE HORA
        try:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M:%S", errors='coerce')
        except:
            # SI FALLA, FALLBACK A DATEUTIL
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
        columns_changes['fecha_to_timestamp'].append(col)
print("[ INFO ] Columnas de fecha convertidas a datetime con formato explícito (fallback a dateutil si falla)")

# 🔧 CAMBIO: NO CONVERTIR COLUMNAS DE HORA A SEGUNDOS — SE MANTIENEN COMO STRING O DATETIME

# 🔧 CAMBIO: INFORMAR QUE NO SE HAN CONVERTIDO HORAS
print("[ INFO ] Columnas de hora mantenidas en su formato original (no convertidas a segundos)")

# DETECTAR COLUMNAS CATEGÓRICAS (TIPO 'OBJECT') PARA CODIFICACIÓN ONE-HOT
cat_cols = df.select_dtypes(include=['object']).columns.tolist()   # IDENTIFICA COLUMNAS DE TEXTO
cat_cols = [c for c in cat_cols if c not in FECHA_COLS]            # EXCLUYE LAS COLUMNAS DE FECHA/HORA

# 🔧 CAMBIO: USAR LABEL ENCODING SIMPLE PARA CATEGORÍAS (EN VEZ DE ONE-HOT)
# Esto reduce dimensionalidad y es más adecuado para LSTM donde se procesan secuencias largas.
from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        columns_changes['one_hot'].append(col)
print("[ INFO ] Label Encoding aplicado en lugar de One-Hot (preserva estructura compacta para LSTM)")

# 🔧 CAMBIO: BLOQUE ORIGINAL ONE-HOT MANTENIDO COMENTADO

# GUARDAR DATASET CODIFICADO EN NUEVO CSV
df.to_csv(OUTPUT_FILE, index=False)                                # EXPORTA EL DATASET LIMPIO Y CODIFICADO
print(f"[ GUARDADO ] Dataset codificado en '{OUTPUT_FILE}'")       # MENSAJE DE CONFIRMACIÓN

# GUARDAR JSON CON REGISTRO DE CAMBIOS REALIZADOS
with open(AUX_FILE, 'w') as f:                                     # ABRE ARCHIVO JSON EN MODO ESCRITURA
    json.dump(columns_changes, f, indent=4)                        # GUARDA LOS CAMBIOS CON FORMATO LEGIBLE
print(f"[ GUARDADO ] Cambios de columnas guardados en '{AUX_FILE}'")  # MENSAJE FINAL DE CONFIRMACIÓN
