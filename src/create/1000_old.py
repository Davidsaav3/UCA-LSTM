# src/main.py

# =========================
# IMPORTS Y CONFIGURACIÓN
# =========================
import os           # PARA MANEJO DE RUTAS Y DIRECTORIOS
import logging      # PARA REGISTRAR INFORMACIÓN DURANTE LA EJECUCIÓN
import pickle       # PARA GUARDAR OBJETOS PYTHON EN ARCHIVOS
import pandas as pd # PARA MANIPULAR DATAFRAMES
import time         # PARA TIMESTAMP Y CONTROL DE TIEMPO
import numpy as np  # PARA OPERACIONES NUMÉRICAS
from datadataset import load_dataset, preprocess_data  # FUNCIONES PERSONALIZADAS PARA DATASET
from model import HybridIFLSTM  # CLASE DEL MODELO HÍBRIDO IF-LSTM

# CONFIGURACIÓN DEL LOGGING (INFORMACIÓN EN CONSOLA)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# =========================
# CONSTANTES Y DIRECTORIOS
# =========================
TIMESTAMP = time.strftime("%Y%m%d_%H%M")  # TIMESTAMP PARA NOMBRAR RESULTADOS
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', TIMESTAMP))
# CREA LA CARPETA DE RESULTADOS SI NO EXISTE
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.info(f"Carpeta de resultados: {RESULTS_DIR}")

# =========================
# FUNCIONES DE GUARDADO
# =========================
def save_csv(df, filename):
    """GUARDA UN DATAFRAME COMO CSV EN LA CARPETA DE RESULTADOS"""
    path = os.path.join(RESULTS_DIR, f"{filename}_{TIMESTAMP}.csv")
    df.to_csv(path)
    logging.info(f"Guardado: {path}")

def save_pkl(obj, filename):
    """GUARDA UN OBJETO PYTHON COMO PICKLE"""
    path = os.path.join(RESULTS_DIR, f"{filename}_{TIMESTAMP}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f"Guardado PKL: {path}")

def save_txt(content, filename):
    """GUARDA TEXTO EN UN ARCHIVO"""
    path = os.path.join(RESULTS_DIR, f"{filename}_{TIMESTAMP}.txt")
    with open(path, 'w') as f:
        f.write(content)
    logging.info(f"Guardado TXT: {path}")

# =========================
# EJECUCIÓN PRINCIPAL
# =========================
if __name__ == "__main__":
    logging.info("Ejecutando pipeline principal.")

    # -------------------------
    # CARGA Y PREPROCESADO
    # -------------------------
    df = load_dataset()  # CARGA EL DATASET ORIGINAL
    logging.info(f"Dataset cargado: {df.shape}.")
    
    df_pre = preprocess_data(df)  # LIMPIA, NORMALIZA Y PREPROCESA EL DATASET
    save_csv(df_pre, 'full_preprocessed')  # GUARDA EL DATASET PREPROCESADO COMPLETO

    # -------------------------
    # INICIALIZACIÓN DEL MODELO HYBRID IF-LSTM
    # -------------------------
    model = HybridIFLSTM(
        seq_length=15,  # LONGITUD DE SECUENCIA PARA LSTM
        lstm_params={'units': 100, 'epochs': 10, 'batch_size': 64},  # PARÁMETROS LSTM
        aggregate_infra=False,           # INDICA SI SE AGRUPAN INFRAESTRUCTURAS
        variance_threshold=0.0,       # CAMBIO: Umbral 0.0 para mantener todas las columnas y evitar eliminaciones sensibles
        diff_threshold=1.0              # UMBRAL PARA DETECCIÓN DE ANOMALÍAS IF
    )

    # -------------------------
    # DIVISIÓN HISTÓRICO / REALTIME
    # -------------------------
    # 80% histórico para entrenamiento, 20% real-time para predicción
    historical_df = df_pre.iloc[:int(0.8 * len(df_pre))]
    realtime_df = df_pre.iloc[int(0.8 * len(df_pre)):]
    save_csv(historical_df, 'historical_preprocessed')
    save_csv(realtime_df, 'realtime_preprocessed')

    # -------------------------
    # PREPARACIÓN DE DATOS PARA IF Y LSTM
    # -------------------------
    # CAMBIO: Procesar historical primero (sin flag, para fit filtro)
    historical_inf_df, context_historical_df = model.load_and_split_data(historical_df)
    save_csv(historical_inf_df, 'historical_inf')        # DATOS DE INFRAESTRUCTURA HISTÓRICA
    save_csv(context_historical_df, 'historical_ctx')   # DATOS CONTEXTUALES HISTÓRICOS

    # CAMBIO: Procesar realtime con use_selected_cols=True para alinear cols al historical
    realtime_inf_df, context_realtime_df = model.load_and_split_data(realtime_df, use_selected_cols=True)
    save_csv(realtime_inf_df, 'realtime_inf')           # DATOS DE INFRAESTRUCTURA REAL-TIME
    save_csv(context_realtime_df, 'realtime_ctx')       # DATOS CONTEXTUALES REAL-TIME

    # CONVIERTE DATAFRAMES A ARRAYS PARA IF Y LSTM
    historical_inf = historical_inf_df.values
    context_historical = context_historical_df.values
    realtime_inf = realtime_inf_df.values
    context_realtime = context_realtime_df.values

    # INDICES PARA MANTENER EL MAPEADO ORIGINAL DE FILAS
    realtime_indices = realtime_inf_df.index

    # -------------------------
    # EJECUCIÓN DE ISOLATION FOREST
    # -------------------------
    model.tune_if(historical_inf)      # AJUSTA PARÁMETROS DEL IF CON DATOS HISTÓRICOS
    model.train_if(historical_inf)     # ENTRENAMIENTO DEL IF
    anomalies = model.run_if(realtime_inf)  # DETECCIÓN DE ANOMALÍAS EN REALTIME
    infrastructure_anomaly = realtime_inf_df[anomalies]  # FILAS CON ANOMALÍAS DETECTADAS
    save_csv(infrastructure_anomaly, 'anomalies_if')

    # -------------------------
    # EJECUCIÓN DE LSTM
    # -------------------------
    try:
        if model.lstm_model is None:
            # ENTRENAMIENTO LSTM EN HILO PARA NO BLOQUEAR
            thread = model.train_lstm_parallel(historical_inf, context_historical)
            thread.join()
            model.update_lstm_model()  # ACTUALIZA EL MODELO LSTM CON RESULTADOS DEL HILO

        preds = model.run_lstm(realtime_inf, context_realtime)  # PREDICCIONES LSTM

        # AJUSTE DE ÍNDICE PARA PREDICCIONES (OFFSET POR seq_length)
        pred_index = realtime_indices[model.seq_length : model.seq_length + len(preds)]
        # CAMBIO: Usa columnas reales de realtime_inf_df post-procesamiento para predicciones
        infrastructure_prediction = pd.DataFrame(preds, columns=realtime_inf_df.columns, index=pred_index)
        save_csv(infrastructure_prediction, 'lstm_predictions')
    except Exception as e:
        logging.error(f"Error en LSTM: {e}")
        # CAMBIO: Usa shape real de historical_inf para fallback vacío
        fallback_dim = historical_inf.shape[1] if 'historical_inf' in locals() and len(historical_inf) > 0 else 0
        preds = np.empty((0, fallback_dim))
        infrastructure_prediction = pd.DataFrame()

    # -------------------------
    # DIAGNÓSTICO DE ANOMALÍAS
    # -------------------------
    diagnostics, fp_mask, fn_mask, confirmed_mask = model.diagnostic(anomalies, realtime_inf, preds)

    # REPORTE RESUMEN
    diag_report = f"FP: {np.sum(fp_mask)}, FN: {np.sum(fn_mask)}, Confirmadas: {np.sum(confirmed_mask)}\nDiagnósticos: {diagnostics[:20]}"
    save_txt(diag_report, 'diagnostics_report')

    # SEPARA ANOMALÍAS SEGÚN TIPO
    confirmed_anomalies = realtime_inf_df[confirmed_mask]
    fp_anomalies = realtime_inf_df[fp_mask]
    fn_anomalies = realtime_inf_df[fn_mask]
    save_csv(confirmed_anomalies, 'anomalies_final')
    save_csv(fp_anomalies, 'fp_anomalies')
    save_csv(fn_anomalies, 'fn_anomalies')

    # FILAS CORRECTAS: NO ANOMALÍAS IF Y NO FN
    infrastructure_correct = realtime_inf_df[~(anomalies | fn_mask)]
    model.supervision(preds, infrastructure_correct.values)

    # -------------------------
    # GUARDADO FINAL DE RESULTADOS (JSON EN LUGAR DE PKL)
    # -------------------------
    import json

    def save_json(obj, filename):
        """GUARDA UN OBJETO COMO JSON"""
        path = os.path.join(RESULTS_DIR, f"{filename}_{TIMESTAMP}.json")
        # CONVERSIÓN SEGURA PARA OBJETOS NO SERIALIZABLES
        def safe_convert(o):
            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient='records')
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.int64, np.float64)):
                return float(o)
            return str(o)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, default=safe_convert, indent=4, ensure_ascii=False)
        logging.info(f"Guardado JSON: {path}")

    results = {
        'anomalies_if': infrastructure_anomaly,
        'confirmed_anomalies': confirmed_anomalies,
        'predictions': infrastructure_prediction,
        'diagnostics': diagnostics
    }
    save_json(results, 'full_results')

    logging.info("Pipeline completado.")
    print("Diagnósticos:", diagnostics[:20])
    print("Anomalías confirmadas head:", confirmed_anomalies.head())