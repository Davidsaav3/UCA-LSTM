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

# INICIALIZAR ARCHIVO AUX PARA REGISTRAR MODIFICACIONES REALIZADAS
columns_changes = {
    'fecha_to_timestamp': [],     # COLUMNAS DE FECHA CONVERTIDAS A TIMESTAMP
    'hora_to_seconds': [],        # COLUMNAS DE HORA CONVERTIDAS A SEGUNDOS
    'one_hot': []                 # COLUMNAS CODIFICADAS MEDIANTE ONE-HOT
}

# CONVERTIR COLUMNAS DE FECHA A TIMESTAMP (SEGUNDOS DESDE 1970)
for col in ['datetime', 'date']:                         # RECORRE COLUMNAS DE FECHA PRINCIPALES
    if col in df.columns:                                # COMPRUEBA QUE EXISTAN EN EL DATASET
        # SE INTENTA CONVERTIR UTILIZANDO FORMATO DÍA/MES/AÑO, HORA MINUTO SEGUNDO SI EXISTE
        try:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M:%S", errors='coerce')  
        except:
            # SI NO COINCIDE EL FORMATO, USAR INFERENCIA AUTOMÁTICA
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce', infer_datetime_format=True)
        # CONVIERTE EL DATETIME A SEGUNDOS DESDE 1970 (UNIX TIMESTAMP)
        df[col] = df[col].astype('int64') // 10**9       
        columns_changes['fecha_to_timestamp'].append(col) # REGISTRA CAMBIO EN EL ARCHIVO AUX
print("[ INFO ] Columnas de fecha convertidas a segundos desde 1970 (incluyendo hora y minuto)")

hora_cols = [c for c in FECHA_COLS if c not in ['datetime', 'date']]  # SELECCIONA COLUMNAS DE HORA EXCLUYENDO FECHAS PRINCIPALES
for col in hora_cols:
    if col in df.columns:
        print(f"[ INFO ] Procesando columna de hora: {col}")  # MOSTRAR NOMBRE DE LA COLUMNA
        # CONVERTIR STRINGS HH:MM A SEGUNDOS DESDE MEDIANOCHE
        df[col] = df[col].astype(str)  # ASEGURA QUE LA COLUMNA SEA TIPO STRING
        # FUNCION AUXILIAR PARA CONVERTIR HORA A SEGUNDOS
        def hora_a_segundos(hora_str):
            try:
                partes = hora_str.split(':')             # SEPARA HORAS Y MINUTOS
                h = int(partes[0])                       # OBTIENE HORAS
                m = int(partes[1])                       # OBTIENE MINUTOS
                return h*3600 + m*60                     # CONVIERTE A SEGUNDOS DESDE MEDIANOCHE
            except:
                return 0                                  # SI EL FORMATO ES INVÁLIDO, DEVUELVE 0
        df[col] = df[col].apply(hora_a_segundos)      # APLICA LA CONVERSIÓN A TODAS LAS FILAS
        columns_changes['hora_to_seconds'].append(col)  # GUARDAR NOMBRE DE LA COLUMNA TRANSFORMADA
print("[ INFO ] Columnas de hora convertidas a SEGUNDOS DESDE MEDIANOCHE")  # MENSAJE FINAL DE CONFIRMACIÓN

# DETECTAR COLUMNAS CATEGÓRICAS (TIPO 'OBJECT') PARA CODIFICACIÓN ONE-HOT
cat_cols = df.select_dtypes(include=['object']).columns.tolist()   # IDENTIFICA COLUMNAS DE TEXTO
cat_cols = [c for c in cat_cols if c not in FECHA_COLS]            # EXCLUYE LAS COLUMNAS DE FECHA/HORA

# APLICAR ONE-HOT ENCODING A LAS COLUMNAS CATEGÓRICAS
for col in cat_cols:                                               # RECORRE CADA COLUMNA CATEGÓRICA
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=ONE_HOT_DROP_FIRST)  
        # CREA VARIABLES DUMMY (ONE-HOT ENCODING) PARA LA COLUMNA CATEGÓRICA
        dummies.columns = [c.replace(" ", REPLACE_SPACES) for c in dummies.columns]   
        # REEMPLAZA ESPACIOS EN LOS NOMBRES DE COLUMNAS POR EL CARÁCTER DEFINIDO EN REPLACE_SPACES
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)                     
        # ELIMINA LA COLUMNA ORIGINAL Y UNE LAS NUEVAS COLUMNAS DUMMY AL DATASET
columns_changes['one_hot'] = cat_cols                              # REGISTRA LAS COLUMNAS TRANSFORMADAS
print("[ INFO ] One-Hot Encoding aplicado (espacios reemplazados por '-')")

# GUARDAR DATASET CODIFICADO EN NUEVO CSV
df.to_csv(OUTPUT_FILE, index=False)                                # EXPORTA EL DATASET LIMPIO Y CODIFICADO
print(f"[ GUARDADO ] Dataset codificado en '{OUTPUT_FILE}'")       # MENSAJE DE CONFIRMACIÓN

# GUARDAR JSON CON REGISTRO DE CAMBIOS REALIZADOS
with open(AUX_FILE, 'w') as f:                                     # ABRE ARCHIVO JSON EN MODO ESCRITURA
    json.dump(columns_changes, f, indent=4)                        # GUARDA LOS CAMBIOS CON FORMATO LEGIBLE
print(f"[ GUARDADO ] Cambios de columnas guardados en '{AUX_FILE}'")  # MENSAJE FINAL DE CONFIRMACIÓN
