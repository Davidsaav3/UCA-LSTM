# src/datadataset.py

# =========================
# IMPORTS Y CONFIGURACIÓN
# =========================
import pandas as pd  # PARA MANEJO DE DATAFRAMES
import numpy as np   # PARA OPERACIONES NUMÉRICAS
import os            # PARA MANEJO DE RUTAS
import logging       # PARA REGISTRAR INFORMACIÓN DURANTE LA EJECUCIÓN

# CONFIGURACIÓN DE LOGGING
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# DESACTIVA OPTIMIZACIONES DE ONE-DNN EN TENSORFLOW (EVITA POSIBLES ERRORES DE COMPUTO)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# =========================
# CARGA DEL DATASET
# =========================
def load_dataset(csv_path='dataset.csv', data_dir=None):
    """
    CARGA UN CSV COMO DATAFRAME DE PANDAS.
    
    ARGUMENTOS:
    - csv_path: NOMBRE DEL CSV A CARGAR (por defecto 'dataset.csv')
    - data_dir: CARPETA DONDE ESTÁ EL CSV (por defecto '../data')
    """
    # SI NO SE ESPECIFICA data_dir, SE USA LA CARPETA ../data
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    # RUTA COMPLETA AL CSV
    full_path = os.path.join(data_dir, csv_path)

    # COMPRUEBA SI EL ARCHIVO EXISTE
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Archivo no encontrado: {full_path}")

    logging.info(f"Cargando dataset desde: {full_path}")

    # CARGA EL CSV EN UN DATAFRAME
    return pd.read_csv(full_path, low_memory=False)


# =========================
# PREPROCESADO DE DATOS
# =========================
def preprocess_data(df):
    """
    PREPROCESA EL DATASET PARA EL MODELO.
    
    PASOS INCLUYEN:
    1. Parseo de datetime
    2. Establecer datetime como índice
    3. Rellenar missing values
    4. Crear features temporales (hora, día de la semana, mes)
    """
    # CREA UNA COPIA PARA NO MODIFICAR EL ORIGINAL
    df = df.copy()
    logging.info("Iniciando preprocesamiento de datos.")

    # -------------------------
    # CREACIÓN DE COLUMNA DATETIME
    # -------------------------
    if 'datetime' not in df.columns and 'date' in df.columns and 'time' in df.columns:
        # COMBINA 'date' + 'time' Y CONVIERTE A DATETIME
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True)
        # ELIMINA COLUMNAS ORIGINALES
        df = df.drop(['date', 'time'], axis=1)
        logging.info("Datetime parseado desde 'date' y 'time'.")

    # -------------------------
    # ESTABLECER DATETIME COMO ÍNDICE
    # -------------------------
    if 'datetime' in df.columns:
        if df.index.name != 'datetime':
            df = df.set_index('datetime').sort_index()
            logging.info("Datetime establecido como índice.")

    # -------------------------
    # RELLENAR MISSING VALUES
    # -------------------------
    # FORWARD FILL Y BACKWARD FILL PARA COMPLETAR NANS
    df = df.ffill().bfill()
    logging.info("Missing values rellenados con ffill/bfill.")

    # -------------------------
    # CREAR FEATURES TEMPORALES
    # -------------------------
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour       # HORA DEL DÍA
        df['weekday'] = df.index.weekday # DÍA DE LA SEMANA (0=Lunes, 6=Domingo)
        df['month'] = df.index.month     # MES
        logging.info("Features temporales derivadas.")

    return df
