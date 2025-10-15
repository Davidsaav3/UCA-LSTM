# src/model.py

# =========================
# IMPORTS Y CONFIGURACIÓN
# =========================
# IMPORTAMOS LIBRERÍAS NECESARIAS PARA PROCESAMIENTO, MODELADO Y LOGGING
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import threading
import queue
import logging
from datadataset import load_dataset  # FUNCION PARA CARGAR DATASET DE EJEMPLO

# CONFIGURAMOS LOGGING
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# =========================
# CLASE HYBRID IF-LSTM
# =========================
class HybridIFLSTM:
    # =========================
    # INICIALIZACIÓN DEL MODELO
    # =========================
    def __init__(self, 
                 infrastructure_cols=None,
                 context_cols=None, 
                 if_params={'n_estimators': 100, 'max_samples': 'auto', 'contamination': 0.1},
                 lstm_params={'units': 50, 'epochs': 20, 'batch_size': 32}, 
                 diff_threshold=1.0, 
                 retrain_threshold=0.1, 
                 seq_length=10,
                 aggregate_infra=True,
                 variance_threshold=1e-5):
        """
        INICIALIZACIÓN DEL MODELO HÍBRIDO IF-LSTM
        """
        # GUARDAMOS LOS PARÁMETROS
        self.if_params = if_params
        self.lstm_params = lstm_params
        self.diff_threshold = diff_threshold
        self.retrain_threshold = retrain_threshold
        self.seq_length = seq_length
        self.aggregate_infra = aggregate_infra
        self.variance_threshold = variance_threshold
        
        logging.info("Inicializando modelo HybridIFLSTM.")
        
        # =========================
        # AUTO-DETECCIÓN DE COLUMNAS SI NO SE PROPORCIONAN
        # =========================
        if infrastructure_cols is None or context_cols is None:
            sample_df = load_dataset()
            all_cols = sample_df.columns.tolist()
            
            # COLUMNAS DE INFRAESTRUCTURA
            self.original_infrastructure_cols = [col for col in all_cols 
                                                 if any(prefix in col for prefix in ['agua_', 'energia_', 'wifi_', 'autoconsumo_', 'fotovolatica_'])] if infrastructure_cols is None else infrastructure_cols
            # COLUMNAS DE CONTEXTO
            self.context_cols = [col for col in all_cols 
                                 if any(prefix in col for prefix in ['clima_', 'aemet_']) or 
                                 col in ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'day_of_year', 
                                         'week_of_year', 'class_day_grado', 'class_day_master', 'working_day', 
                                         'season', 'weekend']] if context_cols is None else context_cols
            
            logging.info(f"Infra cols auto-detectadas: {len(self.original_infrastructure_cols)}, Context: {len(self.context_cols)}.")
        else:
            self.original_infrastructure_cols = infrastructure_cols
            self.context_cols = context_cols
        
        self.infrastructure_cols = self.original_infrastructure_cols
        
        # =========================
        # MODELOS Y SCALERS
        # =========================
        self.if_model = None
        self.lstm_model = None
        self.scaler_inf = StandardScaler()
        self.scaler_ctx = StandardScaler()
        self.scaler_full = StandardScaler()
        self.historical_data = None
        self.correct_data = pd.DataFrame()
        
        # PARA ENTRENAMIENTO PARALLELO DE LSTM
        self.model_queue = queue.Queue()
        self.training_lock = threading.Lock()


    # =========================
    # PREPARACIÓN DE SECUENCIAS PARA LSTM
    # =========================
    def _prepare_sequences(self, data, seq_length):
        """
        PREPARA SECUENCIAS [SAMPLES, TIMESTEPS, FEATURES] PARA LSTM
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)


    # =========================
    # DIVIDIR Y FILTRAR DATOS
    # =========================
    def load_and_split_data(self, df):
        """
        SEPARA INFRAESTRUCTURA Y CONTEXTO, FILTRA COLUMNAS DE BAJA VARIANZA Y AGREGA SI ACTIVADO
        """
        df = df.copy()
        logging.info("Dividiendo datos en infrastructure y context.")
        
        # SELECCIONAMOS COLUMNAS DE INFRAESTRUCTURA Y CONTEXTO
        inf_cols_to_use = [col for col in self.original_infrastructure_cols if col in df.columns]
        inf_data = df[inf_cols_to_use].select_dtypes(include=[np.number]).ffill().bfill()
        ctx_data = df[self.context_cols].select_dtypes(include=[np.number]).ffill().bfill()
        
        # FILTRAMOS COLUMNAS DE BAJA VARIANZA
        if len(inf_data) > 1:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            inf_transformed = selector.fit_transform(inf_data)
            selected_cols = inf_data.columns[selector.get_support()]
            inf_data = pd.DataFrame(inf_transformed, columns=selected_cols, index=inf_data.index)
            logging.info(f"Cols infra después de filtro varianza: {inf_data.shape[1]}.")
        else:
            logging.warning("Datos insuficientes para filtro varianza; saltando.")
        
        # AGREGAMOS COLUMNAS SUMADAS SI ESTÁ ACTIVADO
        if self.aggregate_infra:
            agua_cols = [c for c in inf_data.columns if 'agua_' in c]
            energia_cols = [c for c in inf_data.columns if 'energia_' in c]
            wifi_cols = [c for c in inf_data.columns if 'wifi_' in c]
            auto_cols = [c for c in inf_data.columns if 'autoconsumo_' in c]
            foto_cols = [c for c in inf_data.columns if 'fotovolatica_' in c]
            
            agg_dict = {}
            if agua_cols: agg_dict['total_agua'] = inf_data[agua_cols].sum(axis=1)
            if energia_cols: agg_dict['total_energia'] = inf_data[energia_cols].sum(axis=1)
            if wifi_cols: agg_dict['total_wifi'] = inf_data[wifi_cols].sum(axis=1)
            if auto_cols: agg_dict['total_autoconsumo'] = inf_data[auto_cols].sum(axis=1)
            if foto_cols: agg_dict['total_fotovolatica'] = inf_data[foto_cols].sum(axis=1)
            
            inf_data = pd.DataFrame(agg_dict, index=inf_data.index)
            self.infrastructure_cols = list(inf_data.columns)
            logging.info(f"Cols infra agregadas: {self.infrastructure_cols}.")
        
        return inf_data, ctx_data


    # =========================
    # AJUSTE DE HIPERPARÁMETROS IF
    # =========================
    def tune_if(self, historical_inf):
        """
        TUNNING DE HIPERPARÁMETROS PARA ISOLATION FOREST
        """
        if len(historical_inf) == 0:
            logging.warning("Datos históricos vacíos para IF. Usando params default.")
            return self.if_params
        
        logging.info("Tunneando hiperparámetros de IF.")
        historical_scaled = self.scaler_inf.fit_transform(historical_inf)
        
        best_score = -np.inf
        best_params = self.if_params
        contaminations = [0.05, 0.1, 0.2]
        n_estimators = [50, 100, 200]
        
        for cont in contaminations:
            for est in n_estimators:
                params = {'n_estimators': est, 'contamination': cont, 'max_samples': 'auto'}
                model = IsolationForest(**params, random_state=42)
                scores = model.fit(historical_scaled).decision_function(historical_scaled)
                current_score = np.mean(scores)
                if current_score > best_score:
                    best_score = current_score
                    best_params = params
        
        self.if_params = best_params
        logging.info(f"IF hiperparámetros óptimos: {best_params}")
        return best_params

    # =========================
    # ENTRENAMIENTO IF
    # =========================
    def train_if(self, historical_inf):
        """
        ENTRENA EL MODELO DE ISOLATION FOREST CON DATOS HISTÓRICOS ESCALADOS
        """
        if len(historical_inf) == 0:
            logging.warning("Datos vacíos; saltando entrenamiento IF.")
            return
        
        # ESCALAMOS LOS DATOS
        historical_scaled = self.scaler_inf.transform(historical_inf)
        
        # CREAMOS Y ENTRENAMOS EL MODELO IF
        self.if_model = IsolationForest(**self.if_params, random_state=42)
        self.if_model.fit(historical_scaled)
        logging.info("IF entrenado exitosamente.")


    # =========================
    # DETECCIÓN DE ANOMALÍAS CON IF
    # =========================
    def run_if(self, realtime_inf):
        """
        DETECTA ANOMALÍAS EN DATOS EN TIEMPO REAL CON IF
        """
        if len(realtime_inf) == 0 or self.if_model is None:
            logging.warning("Datos vacíos o IF no entrenado; retornando vacío.")
            return np.array([])
        
        # ESCALAMOS LOS DATOS EN TIEMPO REAL
        realtime_scaled = self.scaler_inf.transform(realtime_inf)
        
        # PREDICCIÓN DE ANOMALÍAS
        labels = self.if_model.predict(realtime_scaled)
        anomalies = labels == -1  # -1 INDICA ANOMALÍA
        logging.info(f"Anomalías detectadas por IF: {np.sum(anomalies)} de {len(anomalies)}.")
        return anomalies


    # =========================
    # ENTRENAMIENTO LSTM EN HILO PARA PARALLEL TRAINING
    # =========================
    def train_lstm_parallel(self, historical_inf, context_historical):
        """
        ENTRENA EL MODELO LSTM EN UN HILO SEPARADO PARA NO BLOQUEAR EL PIPELINE
        """
        def _train():
            with self.training_lock:
                if len(historical_inf) <= self.seq_length:
                    logging.warning("Datos insuficientes para LSTM.")
                    return
                
                # CONCATENAMOS DATOS DE INFRAESTRUCTURA Y CONTEXTO
                full_data = np.hstack([historical_inf, context_historical])
                
                # ESCALAMOS TODOS LOS DATOS
                full_scaled = self.scaler_full.fit_transform(full_data)
                
                # PREPARAMOS SECUENCIAS PARA LSTM
                X, y = self._prepare_sequences(full_scaled, self.seq_length)
                if len(X) == 0:
                    logging.warning("Secuencias insuficientes para LSTM.")
                    return
                
                # TARGET SOLO INFRAESTRUCTURA
                y_inf = y[:, :len(self.infrastructure_cols)]
                
                # DEFINIMOS EL MODELO LSTM
                model = Sequential()
                model.add(Input(shape=(self.seq_length, full_scaled.shape[1])))
                model.add(LSTM(self.lstm_params['units']))
                model.add(Dense(len(self.infrastructure_cols)))
                model.compile(optimizer=Adam(), loss='mse')
                
                # ENTRENAMOS EL MODELO
                model.fit(X, y_inf, epochs=self.lstm_params['epochs'], batch_size=self.lstm_params['batch_size'], verbose=0)
                
                # GUARDAMOS EL MODELO Y SCALER EN LA COLA PARA HITL
                self.model_queue.put((model, self.scaler_full))
                logging.info("LSTM entrenado en paralelo.")
        
        # INICIAMOS EL HILO DE ENTRENAMIENTO
        thread = threading.Thread(target=_train)
        thread.start()
        logging.info("Hilo de entrenamiento LSTM iniciado.")
        return thread


    # =========================
    # ACTUALIZAR MODELO LSTM (HITL)
    # =========================
    def update_lstm_model(self):
        """
        ACTUALIZA EL MODELO LSTM SI EL OPERADOR HUMANO APRUEBA (HITL)
        """
        if not self.model_queue.empty():
            approve = input("[HITL] Aprobar nuevo modelo LSTM? (y/n): ")
            if approve.lower() == 'y':
                self.lstm_model, scaler = self.model_queue.get()
                self.scaler_full = scaler
                logging.info("Modelo LSTM actualizado (aprobado por HITL).")
            else:
                logging.info("Modelo LSTM rechazado (ética HITL).")


    # =========================
    # PREDICCIÓN LSTM
    # =========================
    def run_lstm(self, realtime_inf, context_realtime):
        """
        GENERA PREDICCIONES CON LSTM PARA DATOS EN TIEMPO REAL
        """
        if self.lstm_model is None or len(realtime_inf) <= self.seq_length:
            logging.warning("LSTM no entrenado o datos insuficientes; retornando vacío.")
            return np.empty((0, len(self.infrastructure_cols)))
        
        # CONCATENAMOS INFRAESTRUCTURA Y CONTEXTO
        full_realtime = np.hstack([realtime_inf, context_realtime])
        
        # ESCALAMOS LOS DATOS
        full_scaled = self.scaler_full.transform(full_realtime)
        
        # PREPARAMOS SECUENCIAS
        X, _ = self._prepare_sequences(full_scaled, self.seq_length)
        if len(X) == 0:
            logging.warning("Secuencias insuficientes para predicción LSTM.")
            return np.empty((0, len(self.infrastructure_cols)))
        
        # PREDICCIÓN ESCALADA
        preds_scaled = self.lstm_model.predict(X, verbose=0)
        
        # INVERSIÓN DE ESCALADO
        dummy = np.zeros((len(preds_scaled), full_scaled.shape[1]))
        dummy[:, :len(self.infrastructure_cols)] = preds_scaled
        preds = self.scaler_full.inverse_transform(dummy)[:, :len(self.infrastructure_cols)]
        
        logging.info(f"Predicciones LSTM generadas: {len(preds)} samples.")
        return preds


    # =========================
    # DIAGNÓSTICO DE ANOMALÍAS
    # =========================
    def diagnostic(self, anomalies, realtime_inf, preds):
        """
        COMPARA ANOMALÍAS DETECTADAS POR IF Y DIFERENCIAS CON LSTM
        RETORNA DIAGNÓSTICOS Y MÁSCARAS BOOLEANAS DE FP/FN/CONFIRMADAS
        """
        diagnostics = []
        fp_mask = np.zeros(len(realtime_inf), dtype=bool)
        fn_mask = np.zeros(len(realtime_inf), dtype=bool)
        confirmed_mask = np.zeros(len(realtime_inf), dtype=bool)
        
        offset = self.seq_length
        if len(realtime_inf) < offset or len(preds) == 0:
            logging.warning("Datos insuficientes para diagnóstico; retornando vacío.")
            return diagnostics, fp_mask, fn_mask, confirmed_mask
        
        logging.info("Iniciando diagnóstico de anomalías.")
        for i in range(offset, len(realtime_inf)):
            actual = realtime_inf[i]
            pred = preds[i - offset]
            diff = mean_squared_error(actual, pred)
            is_anomaly_if = anomalies[i]
            
            if is_anomaly_if:
                if diff < self.diff_threshold:
                    diagnostics.append('FP (falso positivo)')
                    fp_mask[i] = True
                else:
                    diagnostics.append('Anomalía confirmada')
                    confirmed_mask[i] = True
            else:
                if diff > self.diff_threshold:
                    diagnostics.append('FN (falso negativo)')
                    fn_mask[i] = True
                else:
                    diagnostics.append('Correcto')
        
        logging.info(f"Diagnósticos completados: {np.sum(fp_mask)} FP, {np.sum(fn_mask)} FN, {np.sum(confirmed_mask)} confirmadas.")
        return diagnostics, fp_mask, fn_mask, confirmed_mask


    # =========================
    # SUPERVISIÓN DE PREDICCIONES
    # =========================
    def supervision(self, preds, realtime_correct):
        """
        SUPERVISIÓN DE PREDICCIONES: CALCULA MSE Y ENRIQUECE HISTÓRICO
        """
        if len(preds) == 0 or len(realtime_correct) == 0:
            logging.warning("Datos insuficientes para supervisión.")
            return
        
        min_len = min(len(realtime_correct), len(preds))
        if min_len == 0:
            logging.warning("No hay datos para calcular MSE en supervisión.")
            return
        
        # CALCULO DEL ERROR
        error = mean_squared_error(realtime_correct[:min_len], preds[:min_len])
        logging.info(f"Error de supervisión (MSE): {error:.4f}")
        
        # RECOMENDACIÓN DE REENTRENAMIENTO
        if error > self.retrain_threshold:
            logging.warning("Umbral excedido: Recomendado reentrenar LSTM.")
        
        # ACTUALIZAMOS HISTÓRICO CON DATOS CORRECTOS
        self.historical_data = pd.concat([self.historical_data, self.correct_data])
        logging.info("Datos históricos actualizados con correctos.")
