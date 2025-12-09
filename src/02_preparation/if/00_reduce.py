import pandas as pd   # MANEJO DE DATAFRAMES
import numpy as np    # OPERACIONES NUMÉRICAS
import os             # MANEJO DE RUTAS Y DIRECTORIOS

# PARÁMETROS
INPUT_FILE = '../../../results/01_dataset/infrastructure_data_realtime.csv'    # RUTA DEL DATASET DE ENTRADA
OUTPUT_FILE = '../../../results/02_preparation/00_reduce.csv'                    # RUTA DEL CSV DE SALIDA
SHOW_INFO = True                                                                 # TRUE = MOSTRAR MENSAJES EN PANTALLA
THRESHOLD_NULLS = 0.3                                                            # PORCENTAJE MÁXIMO DE NULOS PERMITIDOS
THRESHOLD_VARIANCE = 1e-6                                                        # VARIANZA MÍNIMA PARA MANTENER COLUMNA
DATETIME_COL = 'datetime'                                                        # NOMBRE DE COLUMNA DE FECHA/HORA A PROTEGER

# COLUMNAS A ELIMINAR MANUALMENTE
REMOVE_MANUAL = [
    "agua_map00160001",
    "agua_map00330001",
    "fotovolatica_value"
]

# [ CARGAR DATASET ]
df = pd.read_csv(INPUT_FILE, low_memory=False)
if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# [ LIMPIAR NOMBRES DE COLUMNAS ]
df.columns = df.columns.str.strip().str.replace('\ufeff', '')

# [ ASEGURAR COLUMNA DATETIME COMO FECHA ]
if DATETIME_COL in df.columns:
    try:
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    except Exception:
        if SHOW_INFO:
            print(f"[ WARNING ] No se pudo convertir '{DATETIME_COL}' a formato datetime.")

# [ ELIMINAR COLUMNAS NO NUMÉRICAS EXCEPTO FECHAS Y DATETIME ]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
cols_mantener = list(set(numeric_cols + datetime_cols))
if DATETIME_COL in df.columns and DATETIME_COL not in cols_mantener:
    cols_mantener.append(DATETIME_COL)
df = df[cols_mantener]

# [ ELIMINAR COLUMNAS CON DEMASIADOS NULOS, EXCEPTO DATETIME ]
null_ratio = df.isnull().mean()
cols_validas = null_ratio[null_ratio < THRESHOLD_NULLS].index.tolist()
if DATETIME_COL in df.columns and DATETIME_COL not in cols_validas:
    cols_validas.append(DATETIME_COL)
df = df[cols_validas]

# [ ELIMINAR COLUMNAS DE VARIANZA MUY BAJA, EXCEPTO DATETIME ]
varianza = df.var(numeric_only=True)
cols_con_var = varianza[varianza > THRESHOLD_VARIANCE].index.tolist()
if DATETIME_COL in df.columns and DATETIME_COL not in cols_con_var:
    cols_con_var.append(DATETIME_COL)
df = df[cols_con_var]

# [ ELIMINAR COLUMNAS MANUALMENTE ESPECIFICADAS ]
for col in REMOVE_MANUAL:
    if col in df.columns and col != DATETIME_COL:
        df.drop(columns=[col], inplace=True)
        if SHOW_INFO:
            print(f"[ INFO ] Columna '{col}' eliminada manualmente.")

# [ INFORMAR RESULTADOS ]
if SHOW_INFO:
    print(f"[ INFO ] Columnas finales: {df.shape[1]}")
    print(f"[ INFO ] Columnas eliminadas por nulos o baja varianza: {len(null_ratio) - len(cols_con_var)}")

# [ GUARDAR DATASET REDUCIDO ]
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
if SHOW_INFO:
    print(f"[ GUARDADO ] Dataset reducido guardado en '{OUTPUT_FILE}'")
