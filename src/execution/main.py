import subprocess  # EJECUTAR OTROS SCRIPTS DESDE PYTHON
import sys          # USAR EL INTERPRETE ACTUAL DE PYTHON
import logging      # REGISTRAR MENSAJES EN ARCHIVO
import os           # CREAR CARPETAS Y MANEJAR RUTAS

# PARÁMETROS 
RESULTS_FOLDER = '../../results/execution'            # CARPETA DONDE SE GUARDAN RESULTADOS Y LOGS
LOG_FILE = os.path.join(RESULTS_FOLDER, 'log.txt')   # ARCHIVO DE LOG
LOG_LEVEL = logging.INFO                              # NIVEL DE LOG: DEBUG, INFO, WARNING, ERROR
LOG_OVERWRITE = True                                  # TRUE = SOBRESCRIBIR LOG CADA EJECUCIÓN

# LISTA DE SCRIPTS A EJECUTAR EN ORDEN
SCRIPTS = [   
    '01_contaminate.py',             # 1 CONTAMINAR DATASET
    '01_if.py',                      # 1 DETECCIÓN GLOBAL DE ANOMALÍAS

    '02_lstm.py',                    # 2 LSTM 

    '03_diagnostic.py',              # 3 DIAGNOSTICO 

    '04_validation.py'               # 4 VALIDACIÓN DE RESULTADOS
]

# '05_metrics.py',                 # 5 CÁLCULO DE MÉTRICAS
# '05_visualize.py',               # 5 VISUALIZACIÓN DE RESULTADOS

SHOW_OUTPUT = True  # TRUE = IMPRIMIR SALIDA DE LOS SCRIPTS EN PANTALLA

# CREAR CARPETA DE RESULTADOS
os.makedirs(RESULTS_FOLDER, exist_ok=True)  # CREAR CARPETA SI NO EXISTE

# CONFIGURAR LOG
logging.basicConfig(
    filename=LOG_FILE,                     # ARCHIVO DE SALIDA DEL LOG
    filemode='w' if LOG_OVERWRITE else 'a',  # 'w' = SOBRESCRIBIR, 'a' = ADJUNTAR
    level=LOG_LEVEL,                       # NIVEL DE MENSAJES A REGISTRAR
    format='%(message)s',                  # SOLO MENSAJE, SIN FECHA NI NIVEL
    encoding='utf-8'                        # CODIFICACIÓN UTF-8
)

# FUNCIÓN AUXILIAR PARA LOG + PANTALLA
def log_print(msg, level='info'):
    """
    REGISTRAR MENSAJE EN LOG Y OPCIONALMENTE IMPRIMIR EN PANTALLA
    level: 'info' O 'error'
    """
    if level == 'info':
        logging.info(msg)              # GUARDAR MENSAJE EN LOG
        if SHOW_OUTPUT:
            print(msg)                 # IMPRIMIR MENSAJE EN CONSOLA
    elif level == 'error':
        logging.error(msg)             # GUARDAR MENSAJE DE ERROR EN LOG
        if SHOW_OUTPUT:
            print(msg)                 # IMPRIMIR ERROR EN CONSOLA

# EJECUTAR SCRIPTS EN ORDEN
log_print("[ INICIO ]")  # MARCAR INICIO DE LA EJECUCIÓN

for script in SCRIPTS:
    log_print(f"\n[ EJECUTANDO ] {script}\n")  # INFORMAR SCRIPT ACTUAL
    try:
        # LANZAR SCRIPT CON EL INTERPRETE ACTUAL
        process = subprocess.Popen(
            [sys.executable, script],  # USAR PYTHON ACTUAL
            stdout=subprocess.PIPE,    # CAPTURAR SALIDA ESTÁNDAR
            stderr=subprocess.PIPE,    # CAPTURAR ERRORES
            text=True,                 # SALIDA COMO TEXTO
            bufsize=1,                 # BUFFER DE LINEA POR LINEA
            universal_newlines=True    # COMPATIBILIDAD PYTHON 2/3
        )

        # LEER SALIDA ESTÁNDAR LÍNEA POR LÍNEA
        for line in process.stdout:
            log_print(line.rstrip())    # LIMPIAR SALIDA Y LOGUEAR

        # LEER ERRORES LÍNEA POR LÍNEA
        for line in process.stderr:
            log_print(line.rstrip(), level='error')  # LOGUEAR ERRORES

        process.wait()  # ESPERAR QUE EL SCRIPT TERMINE
        if process.returncode != 0:
            log_print(f"[ ERROR ] {script} TERMINÓ CON CÓDIGO {process.returncode}", level='error')

    except Exception as e:
        log_print(f"[ EXCEPCIÓN ] {script}: {e}", level='error')  # CAPTURAR EXCEPCIONES

log_print("\n[ FIN ]")  # MARCAR FIN DE LA EJECUCIÓN
