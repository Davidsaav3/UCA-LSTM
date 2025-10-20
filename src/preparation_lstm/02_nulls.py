import pandas as pd  # PARA MANEJO DE DATAFRAMES
from sklearn.impute import SimpleImputer  # PARA CORREGIR VALORES NULOS
import json  # PARA GUARDAR INFORMACIÓN DE COLUMNAS EN FORMATO JSON

# PARÁMETROS 
INPUT_FILE = '../../data/context_data_realtime.csv'                  # RUTA DEL DATASET ORIGINAL
OUTPUT_CSV = '../../results/preparation/02_nulls.csv'  # RUTA DEL DATASET FINAL SIN NULOS
OUTPUT_JSON = '../../results/preparation/02_aux.json'  # RUTA DEL JSON CON INFORMACIÓN DE COLUMNAS

NUMERIC_STRATEGY = 'median'                            
# ESTRATEGIA PARA RELLENAR NaN EN COLUMNAS NUMÉRICAS
# OPCIONES:
# - 'mean'       : REEMPLAZA CON LA MEDIA. IMPLICA QUE LOS DATOS SE AJUSTAN AL PROMEDIO, PERO PUEDE SER SENSIBLE A OUTLIERS.
# - 'median'     : REEMPLAZA CON LA MEDIANA. RESISTENTE A OUTLIERS Y MANTIENE LA DISTRIBUCIÓN CENTRAL.
# - 'constant'   : REEMPLAZA CON FILL_CONSTANT_NUMERIC. PUEDE INTRODUCIR SESGO SI EL VALOR NO REPRESENTA LA DISTRIBUCIÓN REAL.
# IMPLICA QUE NO QUEDARÁN NaN Y LOS MODELOS PODRÁN PROCESAR TODAS LAS FILAS

FILL_CONSTANT_NUMERIC = 0                               
# VALOR FIJO PARA COLUMNAS NUMÉRICAS SI strategy='constant'
# IMPLICA QUE TODOS LOS NaN SE CONVERTIRÁN EN 0, LO QUE PUEDE INTRODUCIR SESGO SI 0 NO ES REPRESENTATIVO

CATEGORICAL_STRATEGY = 'most_frequent'                 
# ESTRATEGIA PARA RELLENAR NaN EN COLUMNAS CATEGÓRICAS
# OPCIONES:
# - 'most_frequent' : REEMPLAZA CON LA CATEGORÍA MÁS FRECUENTE. PUEDE SESGAR HACIA ESA CATEGORÍA SI HAY MUCHOS NaN.
# - 'constant'      : REEMPLAZA CON FILL_CONSTANT_CATEGORICAL. TRATA LOS NaN COMO NUEVA CATEGORÍA SEPARADA.
# IMPLICA QUE NO HABRÁ NaN Y LAS COLUMNAS PODRÁN SER CODIFICADAS PARA MODELOS

FILL_CONSTANT_CATEGORICAL = 'unknown'                  
# VALOR FIJO PARA COLUMNAS CATEGÓRICAS SI strategy='constant'
# IMPLICA QUE TODOS LOS NaN SE CONVERTIRÁN EN 'UNKNOWN' Y SE TRATARÁN COMO NUEVA CATEGORÍA EN LOS MODELOS

REMOVE_EMPTY_COLUMNS = True                             # ELIMINAR COLUMNAS COMPLETAMENTE VACÍAS
SHOW_INFO = True                                       # MOSTRAR MENSAJES INFORMATIVOS EN PANTALLA
SAVE_INTERMEDIATE = True                              # GUARDAR CSV INTERMEDIO ANTES DE CORREGIR NULOS

# CARGAR DATASET
df = pd.read_csv(INPUT_FILE, low_memory=False)  # LEER EL CSV DE ENTRADA
if SHOW_INFO:
    print(f"[ INFO ] Dataset cargado: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# 🔧 CAMBIO: ORDENAR POR COLUMNA TEMPORAL SI EXISTE (IMPORTANTE PARA LSTM)
if 'datetime' in df.columns:
    df = df.sort_values(by='datetime').reset_index(drop=True)
    if SHOW_INFO:
        print("[ INFO ] Dataset ordenado por columna temporal 'datetime'")

# IDENTIFICAR COLUMNAS POR TIPO
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()  # DETECTAR COLUMNAS NUMÉRICAS
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()  # DETECTAR COLUMNAS CATEGÓRICAS
if SHOW_INFO:
    print(f"[ INFO ] Columnas numéricas: {len(num_cols)}, Columnas categóricas: {len(cat_cols)}")

# GUARDAR INFORMACIÓN INICIAL DE COLUMNAS
columns_info = {
    'categorical': cat_cols,                # LISTA DE COLUMNAS CATEGÓRICAS DETECTADAS
    'numeric_imputed': num_cols,            # COLUMNAS NUMÉRICAS QUE SE CORREGIRAN
    'categorical_imputed': cat_cols,        # COLUMNAS CATEGÓRICAS QUE SE CORREGIRAN
    'removed_empty_columns': []             # COLUMNAS VACÍAS QUE SE ELIMINARÁN
}

# ELIMINAR COLUMNAS COMPLETAMENTE VACÍAS (OPCIONAL)
if REMOVE_EMPTY_COLUMNS:
    empty_cols = df.columns[df.isna().all()].tolist()  # DETECTAR COLUMNAS VACÍAS
    df.drop(columns=empty_cols, inplace=True)          # ELIMINAR COLUMNAS VACÍAS
    columns_info['removed_empty_columns'] = empty_cols  # GUARDAR LISTA DE COLUMNAS ELIMINADAS
    if SHOW_INFO:
        print(f"[ INFO ] Columnas completamente vacías eliminadas: {len(empty_cols)}")

    # 🔧 CAMBIO: ACTUALIZAR LISTAS TRAS ELIMINACIÓN PARA EVITAR ERRORES
    num_cols = [col for col in num_cols if col in df.columns]
    cat_cols = [col for col in cat_cols if col in df.columns]

# GUARDAR DATASET INTERMEDIO (OPCIONAL)
if SAVE_INTERMEDIATE:
    intermediate_csv = OUTPUT_CSV.replace('.csv','_intermediate.csv')  # NOMBRE DEL CSV INTERMEDIO
    df.to_csv(intermediate_csv, index=False)  # GUARDAR DATASET INTERMEDIO
    if SHOW_INFO:
        print(f"[ GUARDADO ] Dataset intermedio en '{intermediate_csv}'")

# 🔧 CAMBIO: IMPUTACIÓN TEMPORAL (EN VEZ DE SimpleImputer GLOBAL)
# Para LSTM se preserva la continuidad temporal rellenando con forward/backward fill
df[num_cols] = df[num_cols].ffill().bfill()  # RELLENAR VALORES NUMÉRICOS HACIA ADELANTE Y ATRÁS
if SHOW_INFO:
    print("[ INFO ] Valores nulos numéricos imputados con 'ffill/bfill' (preserva continuidad temporal)")

df[cat_cols] = df[cat_cols].ffill().bfill()  # RELLENAR CATEGÓRICOS HACIA ADELANTE Y ATRÁS
if SHOW_INFO:
    print("[ INFO ] Valores nulos categóricos imputados con 'ffill/bfill' (preserva secuencia temporal)")

# 🔧 CAMBIO: ELIMINAMOS USO DE SimpleImputer

# GUARDAR DATASET FINAL
df.to_csv(OUTPUT_CSV, index=False)  # GUARDAR CSV FINAL SIN NULOS
if SHOW_INFO:
    print(f"[ GUARDADO ] Dataset final sin nulos en '{OUTPUT_CSV}'")
    print(f"[ INFO ] Dataset final: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS")

# GUARDAR INFORMACIÓN DE COLUMNAS EN JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(columns_info, f, indent=4)  # GUARDAR COLUMNAS CORREGIDAS Y ELIMINADAS
if SHOW_INFO:
    print(f"[ GUARDADO ] Columnas CORREGIDAS guardadas en '{OUTPUT_JSON}'")
