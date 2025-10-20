import pandas as pd  # PARA MANEJO DE DATAFRAMES 
from sklearn.feature_selection import VarianceThreshold  # PARA ELIMINAR COLUMNAS CON BAJA VARIANZA
from sklearn.preprocessing import StandardScaler  # IMPORTADO PERO NO USADO AQUÍ, POSIBLEMENTE PARA ESCALADO FUTURO
import json  

# CONFIGURACIÓN DE ARCHIVOS Y PARÁMETROS
INPUT_CSV = '../../../results/02_preparation/04_scale.csv'        # DATASET DE ENTRADA ESCALADO
OUTPUT_CSV = '../../../results/02_preparation/05_variance.csv'    # DATASET FINAL TRAS ELIMINAR COLUMNAS CON VARIANZA BAJA
OUTPUT_JSON = '../../../results/02_preparation/05_aux.json'       # JSON CON LISTA DE COLUMNAS SELECCIONADAS Y EXCLUIDAS

VAR_THRESHOLD = 0.01            # UMBRAL DE VARIANZA: COLUMNAS CON VAR < 0.01 SERÁN ELIMINADAS

DROP_CONSTANT_COLUMNS = True     # TRUE = ELIMINAR COLUMNAS CON VARIANZA EXACTAMENTE 0
DROP_LOW_VARIANCE = True         # TRUE = ELIMINAR COLUMNAS CON VARIANZA < VAR_THRESHOLD

NUMERIC_ONLY = True              # TRUE = SOLO SE ANALIZAN COLUMNAS NUMÉRICAS

# CARGAR DATASET
df = pd.read_csv(INPUT_CSV, low_memory=False)  # LEER EL CSV DE ENTRADA
if NUMERIC_ONLY:
    # CONSERVAR SOLO COLUMNAS NUMÉRICAS PARA EL ANÁLISIS DE VARIANZA
    df = df.select_dtypes(include=['int64','float64'])
print(f"[ INFO ] DATASET CARGADO: {df.shape[0]} FILAS, {df.shape[1]} COLUMNAS NUMÉRICAS")

# 🔹 MODIFICADO PARA LSTM → Mantener orden temporal si existe columna datetime
original_df = pd.read_csv(INPUT_CSV, low_memory=False)  # 🔹 MODIFICADO PARA LSTM
if 'datetime' in original_df.columns:
    datetime_col = original_df['datetime']  # 🔹 MODIFICADO PARA LSTM
    df = df.assign(datetime=datetime_col)   # 🔹 MODIFICADO PARA LSTM
    df = df.sort_values('datetime')         # 🔹 MODIFICADO PARA LSTM
    df = df.drop(columns=['datetime'])      # 🔹 MODIFICADO PARA LSTM
    if df.shape[0] > 0:
        print("[ INFO ] DATASET ORDENADO POR 'datetime' ANTES DE SELECCIÓN POR VARIANZA")  # 🔹 MODIFICADO PARA LSTM

# 🔧 MODIFICACIÓN: Eliminar columnas con todos NaN antes de aplicar VarianceThreshold
cols_all_nan = df.columns[df.isna().all()].tolist()  # 🔧 DETECTAR COLUMNAS COMPLETAMENTE VACÍAS
if len(cols_all_nan) > 0:
    if SHOW_INFO := True:
        print(f"[ AVISO ] Columnas completamente NaN que se eliminarán antes de la selección por varianza: {cols_all_nan}")  # 🔧 INFO
    df = df.drop(columns=cols_all_nan)  # 🔧 ELIMINAR COLUMNAS TODO NaN

# SELECCIÓN DE COLUMNAS POR VARIANZA
if DROP_CONSTANT_COLUMNS or DROP_LOW_VARIANCE:
    # CREAR OBJETO VarianceThreshold CON EL UMBRAL DEFINIDO
    selector = VarianceThreshold(threshold=VAR_THRESHOLD)
    
    # APLICAR SELECCIÓN DE COLUMNAS CON VARIANZA >= VAR_THRESHOLD
    df_selected = pd.DataFrame(
        selector.fit_transform(df),  # TRANSFORMAR DATAFRAME ELIMINANDO COLUMNAS DE BAJA VARIANZA
        columns=df.columns[selector.get_support()]  # CONSERVAR LOS NOMBRES DE COLUMNAS SELECCIONADAS
    )
    
    # OBTENER COLUMNAS EXCLUIDAS (VAR < UMBRAL)
    excluded_cols = list(df.columns[~selector.get_support()])
    
    # INFORMACIÓN SOBRE RESULTADO DE LA SELECCIÓN
    print(f"[ INFO ] COLUMNAS SELECCIONADAS: {df_selected.shape[1]} / {df.shape[1]}")
    print(f"[ INFO ] COLUMNAS EXCLUIDAS: {len(excluded_cols)}")
else:
    # SI NO SE ELIMINA NINGUNA COLUMNA, COPIAR EL DATAFRAME ORIGINAL
    df_selected = df.copy()
    excluded_cols = []

# 🔹 MODIFICADO PARA LSTM → Reordenar columnas si es necesario
df_selected = df_selected.reset_index(drop=True)  # 🔹 MODIFICADO PARA LSTM

# GUARDAR DATASET FINAL CON COLUMNAS SELECCIONADAS
df_selected.to_csv(OUTPUT_CSV, index=False)
print(f"[ GUARDADO ] DATASET FINAL EN '{OUTPUT_CSV}'")

# GUARDAR INFORMACIÓN DE COLUMNAS SELECCIONADAS Y EXCLUIDAS EN JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump({
        'selected_columns': df_selected.columns.tolist(),  # LISTA DE COLUMNAS MANTENIDAS
        'excluded_columns': excluded_cols                 # LISTA DE COLUMNAS ELIMINADAS
    }, f, indent=4)
print(f"[ GUARDADO ] INFORMACIÓN DE COLUMNAS EN JSON EN '{OUTPUT_JSON}'")

# MENSAJE FINAL
print("[ INFO ] PROCESO DE SELECCIÓN POR VARIANZA FINALIZADO CON ÉXITO")
