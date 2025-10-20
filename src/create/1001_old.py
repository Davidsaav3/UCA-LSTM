# =========================
# IMPORTS Y CONFIGURACIÓN
# =========================
import numpy as np  # BIBLIOTECA PARA OPERACIONES NUMÉRICAS EFICIENTES Y MANEJO DE ARRAYS MULTIDIMENSIONALES
import pandas as pd  # BIBLIOTECA PARA MANIPULACIÓN DE DATOS TABULARES, USANDO DATAFRAMES Y SERIES PARA ANÁLISIS
from sklearn.ensemble import IsolationForest  # ALGORITMO DE DETECCIÓN DE ANOMALÍAS NO SUPERVISADA BASADO EN ENSAMBLES DE ÁRBOLES DE AISLAMIENTO
from sklearn.preprocessing import StandardScaler  # CLASE PARA ESTANDARIZAR CARACTERÍSTICAS ELIMINANDO LA MEDIA Y ESCALANDO A VARIANZA UNITARIA
from sklearn.metrics import mean_squared_error  # FUNCIÓN PARA CALCULAR EL ERROR CUADRÁTICO MEDIO ENTRE VALORES REALES Y PREDICHOS
from sklearn.feature_selection import VarianceThreshold  # SELECTOR QUE ELIMINA CARACTERÍSTICAS CON VARIANZA POR DEBAJO DE UN UMBRAL ESPECÍFICO
from tensorflow.keras.models import Sequential  # MODELO DE RED NEURONAL SECUENCIAL EN KERAS, PARA CONSTRUIR CAPAS LINEALMENTE
from tensorflow.keras.layers import Input, LSTM, Dense  # CAPAS: INPUT PARA DEFINIR ENTRADA, LSTM PARA PROCESAMIENTO DE SECUENCIAS TEMPORALES, DENSE PARA CAPAS FULLY CONNECTED
from tensorflow.keras.optimizers import Adam  # OPTIMIZADOR BASADO EN DESCENSO DE GRADIENTE ADAPTATIVO PARA ENTRENAMIENTO DE REDES
import threading  # MÓDULO PARA CREAR Y MANEJAR HILOS DE EJECUCIÓN, PERMITIENDO OPERACIONES PARALELAS
import queue  # IMPLEMENTACIÓN DE COLAS SEGURAS PARA HILOS, USADA PARA COMUNICACIÓN ENTRE PROCESOS
import logging  # SISTEMA DE REGISTRO DE EVENTOS, ERRORES Y MENSAJES DE DEPURACIÓN CON NIVELES DE SEVERIDAD
from datadataset import load_dataset  # FUNCIÓN PERSONALIZADA PARA CARGAR EL DATASET DE DATOS, POSIBLEMENTE DESDE ARCHIVO O BASE DE DATOS

# CONFIGURACIÓN GLOBAL DE LOGGING: NIVEL INFO PARA MENSAJES INFORMATIVOS, FORMATO CON NIVEL Y MENSAJE
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class HybridIFLSTM:
    # CLASE QUE IMPLEMENTA UN MODELO HÍBRIDO COMBINANDO ISOLATION FOREST (IF) PARA DETECCIÓN DE ANOMALÍAS Y LSTM PARA PREDICCIÓN TEMPORAL
    def __init__(self, 
                 infrastructure_cols=None,  # PARÁMETRO OPCIONAL: LISTA DE COLUMNAS RELACIONADAS CON INFRAESTRUCTURA (EJ. SENSORES DE AGUA, ENERGÍA)
                 context_cols=None,  # PARÁMETRO OPCIONAL: LISTA DE COLUMNAS DE CONTEXTO EXTERNO (EJ. DATOS CLIMÁTICOS, TEMPORALES)
                 if_params={'n_estimators': 100, 'max_samples': 'auto', 'contamination': 0.1},  # DICCIONARIO DE HIPERPARÁMETROS PARA EL MODELO ISOLATION FOREST
                 lstm_params={'units': 50, 'epochs': 20, 'batch_size': 32},  # DICCIONARIO DE HIPERPARÁMETROS PARA LA RED LSTM (UNIDADES, ÉPOCAS, TAMAÑO DE LOTE)
                 diff_threshold=1.0,  # UMBRAL DE DIFERENCIA (BASADO EN MSE) PARA VALIDAR ANOMALÍAS ENTRE IF Y LSTM
                 retrain_threshold=0.1,  # UMBRAL DE ERROR (MSE) PARA DECIDIR SI REENTRENAR EL MODELO LSTM
                 seq_length=10,  # LONGITUD DE LAS SECUENCIAS TEMPORALES USADAS EN LA LSTM PARA CAPTURAR DEPENDENCIAS TEMPORALES
                 aggregate_infra=True,  # BANDERA BOOLEANA: SI AGREGAR COLUMNAS DE INFRAESTRUCTURA POR CATEGORÍAS (SUMA TOTALES)
                 variance_threshold=1e-5):  # UMBRAL MÍNIMO DE VARIANZA PARA FILTRAR CARACTERÍSTICAS IRRELEVANTES O CONSTANTES
        # ASIGNACIÓN DE PARÁMETROS DE INICIALIZACIÓN A ATRIBUTOS DE LA INSTANCIA PARA USO POSTERIOR
        self.if_params = if_params
        self.lstm_params = lstm_params
        self.diff_threshold = diff_threshold
        self.retrain_threshold = retrain_threshold
        self.seq_length = seq_length
        self.aggregate_infra = aggregate_infra
        self.variance_threshold = variance_threshold
        
        # REGISTRO EN LOG DEL INICIO DEL PROCESO DE INICIALIZACIÓN DEL MODELO HÍBRIDO
        logging.info("Inicializando modelo HybridIFLSTM.")
        
        # LÓGICA PARA DETECCIÓN AUTOMÁTICA DE COLUMNAS SI NO SE PROPORCIONAN EXPLÍCITAMENTE
        if infrastructure_cols is None or context_cols is None:
            sample_df = load_dataset()  # CARGA DE UN DATASET DE MUESTRA PARA ANALIZAR ESTRUCTURA DE COLUMNAS
            all_cols = sample_df.columns.tolist()  # OBTENCIÓN DE LISTA DE TODAS LAS COLUMNAS DEL DATASET
            # SELECCIÓN AUTOMÁTICA DE COLUMNAS DE INFRAESTRUCTURA BASADA EN PREFIJOS ESPECÍFICOS (EJ. 'agua_', 'energia_')
            self.original_infrastructure_cols = [col for col in all_cols 
                                                 if any(prefix in col for prefix in ['agua_', 'energia_', 'wifi_', 'autoconsumo_', 'fotovolatica_'])] if infrastructure_cols is None else infrastructure_cols
            # SELECCIÓN AUTOMÁTICA DE COLUMNAS DE CONTEXTO BASADA EN PREFIJOS Y CAMPOS TEMPORALES/CLIMÁTICOS PREDEFINIDOS
            self.context_cols = [col for col in all_cols 
                                 if any(prefix in col for prefix in ['clima_', 'aemet_']) or 
                                 col in ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'day_of_year', 
                                         'week_of_year', 'class_day_grado', 'class_day_master', 'working_day', 
                                         'season', 'weekend']] if context_cols is None else context_cols
            # REGISTRO EN LOG DEL NÚMERO DE COLUMNAS DETECTADAS AUTOMÁTICAMENTE PARA INFRAESTRUCTURA Y CONTEXTO
            logging.info(f"Infra cols auto-detectadas: {len(self.original_infrastructure_cols)}, Context: {len(self.context_cols)}.")
        else:
            # USO DE COLUMNAS PROPORCIONADAS MANUALMENTE SI ESTÁN ESPECIFICADAS EN LOS PARÁMETROS
            self.original_infrastructure_cols = infrastructure_cols
            self.context_cols = context_cols
        
        # COMENTARIO EXPLÍCATIVO: NO SE ASIGNAN infrastructure_cols AQUÍ PARA MANTENER CONSISTENCIA POST-FILTRADO EN MÉTODOS POSTERIORES
        
        # INICIALIZACIÓN DE ATRIBUTOS PARA MODELOS Y PREPROCESADORES
        self.if_model = None  # MODELO ISOLATION FOREST, INICIALMENTE NONE, SE ENTRENARÁ MÁS TARDE
        self.lstm_model = None  # MODELO LSTM, INICIALMENTE NONE, SE ENTRENARÁ EN HILO PARALELO
        self.scaler_inf = StandardScaler()  # ESCALADOR DEDICADO PARA DATOS DE INFRAESTRUCTURA
        self.scaler_ctx = StandardScaler()  # ESCALADOR DEDICADO PARA DATOS DE CONTEXTO (AUNQUE NO SIEMPRE USADO DIRECTAMENTE)
        self.scaler_full = StandardScaler()  # ESCALADOR PARA DATOS COMBINADOS (INFRA + CONTEXTO) EN EL FLUJO LSTM
        self.historical_data = None  # ALMACÉN PARA DATOS HISTÓRICOS ACUMULADOS DURANTE LA EJECUCIÓN
        self.correct_data = pd.DataFrame()  # DATAFRAME VACÍO PARA ALMACENAR DATOS CONFIRMADOS COMO CORRECTOS
        self.model_queue = queue.Queue()  # COLA SEGURA PARA HILOS DONDE SE DEPOSITARÁ EL MODELO LSTM ENTRENADO
        self.training_lock = threading.Lock()  # BLOQUEO DE HILO PARA EVITAR CONDICIONES DE CARRERA DURANTE ENTRENAMIENTO PARALELO
        
        # VARIABLES AUXILIARES PARA MANTENER CONSISTENCIA EN SELECCIÓN DE COLUMNAS ENTRE LOTES DE DATOS
        self.selected_pre_agg_cols = None  # COLUMNAS SELECCIONADAS ANTES DE LA AGREGACIÓN (POST-FILTRO DE VARIANZA)
        self.final_infrastructure_cols = None  # COLUMNAS FINALES DE INFRAESTRUCTURA DESPUÉS DE FILTROS Y AGREGACIÓN

    def _prepare_sequences(self, data, seq_length):
        # MÉTODO PRIVADO: CONSTRUYE SECUENCIAS DESLIZANTES PARA ENTRENAMIENTO O PREDICCIÓN EN MODELOS TEMPORALES COMO LSTM
        # RECIBE DATOS (ARRAY 2D) Y LONGITUD DE SECUENCIA, GENERA X (VENTANAS DE ENTRADA) E Y (VALORES SIGUIENTES)
        X, y = [], []  # LISTAS TEMPORALES PARA ALMACENAR SECUENCIAS DE ENTRADA Y OBJETIVOS
        for i in range(len(data) - seq_length):  # BUCLE PARA CREAR VENTANAS DESDE EL INICIO HASTA EL FINAL MENOS LA LONGITUD
            X.append(data[i:i + seq_length])  # AÑADE VENTANA DE SECUENCIA COMO ENTRADA
            y.append(data[i + seq_length])  # AÑADE EL VALOR INMEDIATAMENTE POSTERIOR COMO OBJETIVO
        return np.array(X), np.array(y)  # CONVERSIÓN A ARRAYS NUMPY PARA COMPATIBILIDAD CON TENSORFLOW/KERAS

    def load_and_split_data(self, df, use_selected_cols=False):
        # MÉTODO: PROESA EL DATAFRAME DE ENTRADA, DIVIDE EN INFRAESTRUCTURA Y CONTEXTO, APLICA FILTROS Y RELLENOS
        # OPCIONALMENTE USA COLUMNAS PREVIAMENTE SELECCIONADAS PARA CONSISTENCIA EN MODO TIEMPO REAL
        df = df.copy()  # CREA COPIA DEL DATAFRAME PARA EVITAR MODIFICACIONES EN EL ORIGINAL Y POSIBLES EFECTOS SECUNDARIOS
        logging.info("Dividiendo datos en infrastructure y context.")  # REGISTRO EN LOG DEL INICIO DE LA DIVISIÓN DE DATOS
        
        # FILTRA COLUMNAS DE INFRAESTRUCTURA PRESENTES EN EL DATAFRAME ACTUAL
        inf_cols_to_use = [col for col in self.original_infrastructure_cols if col in df.columns]
        # EXTRAE DATOS NUMÉRICOS DE INFRAESTRUCTURA, RELLENA VALORES FALTANTES CON PROPAGACIÓN HACIA ADELANTE Y ATRÁS
        inf_data = df[inf_cols_to_use].select_dtypes(include=[np.number]).ffill().bfill()
        # LO MISMO PARA COLUMNAS DE CONTEXTO
        ctx_data = df[self.context_cols].select_dtypes(include=[np.number]).ffill().bfill()
        
        original_cols = inf_data.columns.tolist()  # GUARDA LISTA DE COLUMNAS ORIGINALES ANTES DE CUALQUIER FILTRADO
        
        # APLICACIÓN DE FILTRO DE VARIANZA PARA ELIMINAR COLUMNAS CON POCA VARIABILIDAD (IRRELEVANTES)
        if len(inf_data) > 1:  # VERIFICA SI HAY SUFICIENTES FILAS PARA CALCULAR VARIANZA DE MANERA SIGNIFICATIVA
            selector = VarianceThreshold(threshold=self.variance_threshold)  # INSTANCIA EL SELECTOR CON UMBRAL DEFINIDO
            
            if use_selected_cols and self.selected_pre_agg_cols is not None:  # MODO PARA DATOS EN TIEMPO REAL: ALINEA CON SELECCIÓN PREVIA
                missing_cols = [col for col in self.selected_pre_agg_cols if col not in inf_data.columns]  # IDENTIFICA COLUMNAS FALTANTES
                for col in missing_cols:  # AÑADE COLUMNAS FALTANTES LLENAS DE CEROS PARA MANTENER DIMENSIONALIDAD
                    inf_data[col] = 0
                inf_data = inf_data[self.selected_pre_agg_cols]  # RESTRINGE A LAS COLUMNAS PREVIAMENTE SELECCIONADAS
                logging.info(f"Cols infra en realtime alineadas pre-agregado: {inf_data.shape[1]}.")  # LOG DE ALINEACIÓN EN TIEMPO REAL
            else:  # MODO ENTRENAMIENTO: APLICA EL FILTRO DE VARIANZA PARA SELECCIONAR CARACTERÍSTICAS
                inf_transformed = selector.fit_transform(inf_data)  # TRANSFORMA LOS DATOS ELIMINANDO BAJA VARIANZA
                selected_cols = inf_data.columns[selector.get_support()]  # OBTIENE NOMBRES DE COLUMNAS QUE SOBREVIVIERON AL FILTRO
                inf_data = pd.DataFrame(inf_transformed, columns=selected_cols, index=inf_data.index)  # RECONSTRUYE DATAFRAME FILTRADO
                logging.info(f"Cols infra después de filtro varianza: {inf_data.shape[1]}.")  # LOG DEL NÚMERO DE COLUMNAS POST-FILTRO
                
                self.selected_pre_agg_cols = selected_cols.tolist()  # ALMACENA COLUMNAS SELECCIONADAS PARA USO FUTURO EN TIEMPO REAL
                
                removed = [col for col in original_cols if col not in selected_cols]  # CALCULA COLUMNAS ELIMINADAS
                if removed:  # SI HUBO ELIMINACIONES, REGISTRA EN LOG PARA DEPURACIÓN
                    logging.info(f"Columnas removidas por baja varianza en este set: {removed}")
        else:
            # MANEJO DE CASO CON DATOS INSUFICIENTES: ADVERTENCIA Y USO DE COLUMNAS ORIGINALES SIN FILTRO
            logging.warning("Datos insuficientes para filtro varianza; usando cols originales.")
            if use_selected_cols and self.selected_pre_agg_cols is not None:  # AUN ASÍ, ALINEA EN MODO REALTIME SI POSIBLE
                missing_cols = [col for col in self.selected_pre_agg_cols if col not in inf_data.columns]
                for col in missing_cols:
                    inf_data[col] = 0
                inf_data = inf_data[self.selected_pre_agg_cols]
        
        # ACTUALIZACIÓN DE ATRIBUTOS DE COLUMNAS POST-FILTRO, SOLO SI NO ES MODO REALTIME (PARA HISTÓRICO)
        if not use_selected_cols:
            self.infrastructure_cols = inf_data.columns.tolist()
            self.final_infrastructure_cols = self.infrastructure_cols
        
        # AGREGACIÓN DE COLUMNAS DE INFRAESTRUCTURA POR CATEGORÍAS SI LA OPCIÓN ESTÁ ACTIVADA, PARA REDUCIR DIMENSIONALIDAD
        if self.aggregate_infra:
            agg_keys = ['total_agua', 'total_energia', 'total_wifi', 'total_autoconsumo', 'total_fotovolatica']  # DEFINICIÓN DE CLAVES PARA TOTALES AGREGADOS
            agg_dict = {key: pd.Series(0, index=inf_data.index) for key in agg_keys}  # INICIALIZA DICCIONARIO CON SERIES VACÍAS
            
            # IDENTIFICACIÓN DE COLUMNAS POR PREFIJO PARA CADA CATEGORÍA
            agua_cols = [c for c in inf_data.columns if 'agua_' in c]
            energia_cols = [c for c in inf_data.columns if 'energia_' in c]
            wifi_cols = [c for c in inf_data.columns if 'wifi_' in c]
            auto_cols = [c for c in inf_data.columns if 'autoconsumo_' in c]
            foto_cols = [c for c in inf_data.columns if 'fotovolatica_' in c]
            
            # SUMA HORIZONTAL DE COLUMNAS POR CATEGORÍA SI EXISTEN
            if agua_cols: agg_dict['total_agua'] = inf_data[agua_cols].sum(axis=1)
            if energia_cols: agg_dict['total_energia'] = inf_data[energia_cols].sum(axis=1)
            if wifi_cols: agg_dict['total_wifi'] = inf_data[wifi_cols].sum(axis=1)
            if auto_cols: agg_dict['total_autoconsumo'] = inf_data[auto_cols].sum(axis=1)
            if foto_cols: agg_dict['total_fotovolatica'] = inf_data[foto_cols].sum(axis=1)
            
            inf_data = pd.DataFrame(agg_dict, index=inf_data.index)  # RECONSTRUYE DATAFRAME CON SOLO LAS COLUMNAS AGREGADAS
            
            if not use_selected_cols:  # ACTUALIZA COLUMNAS FINALES EN MODO ENTRENAMIENTO
                self.final_infrastructure_cols = list(inf_data.columns)
                self.infrastructure_cols = self.final_infrastructure_cols
            else:  # EN MODO REALTIME, REINDEXA PARA ALINEAR CON COLUMNAS FINALES PREVIAS, RELLENANDO CON CEROS
                inf_data = inf_data.reindex(columns=self.final_infrastructure_cols, fill_value=0)
            
            logging.info(f"Cols infra agregadas y alineadas: {self.infrastructure_cols}.")  # LOG DE ESTADO POST-AGREGACIÓN
        
        # RETORNO DE DATOS PROCESADOS: INFRAESTRUCTURA Y CONTEXTO LISTOS PARA MODELOS
        return inf_data, ctx_data

    def tune_if(self, historical_inf):
        # MÉTODO PARA AJUSTAR HIPERPARÁMETROS DEL ISOLATION FOREST USANDO DATOS HISTÓRICOS MEDIANTE BÚSQUEDA EN GRID SIMPLE
        if len(historical_inf) == 0:  # MANEJO DE CASO VACÍO: ADVERTENCIA Y RETORNO DE PARÁMETROS DEFAULT
            logging.warning("Datos históricos vacíos para IF. Usando params default.")
            return self.if_params
        
        logging.info("Tunneando hiperparámetros de IF.")  # LOG DE INICIO DEL PROCESO DE TUNING
        
        historical_scaled = self.scaler_inf.fit_transform(historical_inf)  # ESCALADO DE DATOS HISTÓRICOS PARA MEJOR RENDIMIENTO DEL MODELO
        
        best_score = -np.inf  # INICIALIZACIÓN DEL MEJOR PUNTAJE (SE MAXIMIZA EL PROMEDIO DE SCORES DE DECISIÓN)
        best_params = self.if_params  # PARÁMETROS INICIALES COMO BASE
        contaminations = [0.05, 0.1, 0.2]  # RANGO DE VALORES PARA 'CONTAMINATION' (PROPORCIÓN ESPERADA DE ANOMALÍAS)
        n_estimators = [50, 100, 200]  # RANGO DE VALORES PARA NÚMERO DE ÁRBOLES EN EL ENSAMBLE
        
        # BÚSQUEDA EXHAUSTIVA EN COMBINACIONES DE PARÁMETROS
        for cont in contaminations:
            for est in n_estimators:
                params = {'n_estimators': est, 'contamination': cont, 'max_samples': 'auto'}  # CONSTRUCCIÓN DE DICCIONARIO DE PARÁMETROS TEMPORAL
                model = IsolationForest(**params, random_state=42)  # INSTANCIACIÓN DE MODELO TEMPORAL PARA EVALUACIÓN
                scores = model.fit(historical_scaled).decision_function(historical_scaled)  # ENTRENAMIENTO Y OBTENCIÓN DE SCORES
                current_score = np.mean(scores)  # CÁLCULO DE PUNTAJE PROMEDIO (MAYOR VALOR INDICA MEJOR SEPARACIÓN)
                if current_score > best_score:  # ACTUALIZACIÓN SI SE ENCUENTRA MEJOR COMBINACIÓN
                    best_score = current_score
                    best_params = params
        
        self.if_params = best_params  # ASIGNACIÓN DE MEJORES PARÁMETROS A LA INSTANCIA
        logging.info(f"IF hiperparámetros óptimos: {best_params}")  # LOG DE RESULTADOS ÓPTIMOS
        return best_params  # RETORNO PARA USO EXTERNO SI NECESARIO

    def train_if(self, historical_inf):
        # MÉTODO PARA ENTRENAR EL MODELO ISOLATION FOREST CON LOS PARÁMETROS AJUSTADOS O DEFAULT
        if len(historical_inf) == 0:  # MANEJO DE DATOS VACÍOS: ADVERTENCIA Y SALTO DE ENTRENAMIENTO
            logging.warning("Datos vacíos; saltando entrenamiento IF.")
            return
        
        historical_scaled = self.scaler_inf.transform(historical_inf)  # ESCALADO USANDO SCALER AJUSTADO PREVIAMENTE
        
        self.if_model = IsolationForest(**self.if_params, random_state=42)  # INSTANCIACIÓN DEL MODELO CON PARÁMETROS
        self.if_model.fit(historical_scaled)  # ENTRENAMIENTO PROPIAMENTE DICHO EN DATOS ESCALADOS
        logging.info("IF entrenado exitosamente.")  # LOG DE ENTRENAMIENTO COMPLETADO

    def run_if(self, realtime_inf):
        # MÉTODO PARA EJECUTAR DETECCIÓN DE ANOMALÍAS EN DATOS EN TIEMPO REAL USANDO EL MODELO IF ENTRENADO
        if len(realtime_inf) == 0 or self.if_model is None:  # VERIFICACIÓN DE CONDICIONES INVÁLIDAS: DATOS O MODELO AUSENTES
            logging.warning("Datos vacíos o IF no entrenado; retornando vacío.")
            return np.array([])
        
        # VERIFICACIÓN DE COMPATIBILIDAD DE DIMENSIONES ENTRE ENTRENAMIENTO Y EJECUCIÓN
        if realtime_inf.shape[1] != self.scaler_inf.n_features_in_:
            logging.error(f"Shape mismatch en run_if: expected {self.scaler_inf.n_features_in_}, got {realtime_inf.shape[1]}")  # REGISTRO DE ERROR DE DIMENSIÓN
            # AJUSTE MANUAL AÑADIENDO COLUMNAS DE CEROS PARA ALINEAR DIMENSIONES
            realtime_inf = np.hstack([realtime_inf, np.zeros((len(realtime_inf), self.scaler_inf.n_features_in_ - realtime_inf.shape[1]))])
        
        realtime_scaled = self.scaler_inf.transform(realtime_inf)  # ESCALADO DE DATOS EN TIEMPO REAL
        labels = self.if_model.predict(realtime_scaled)  # PREDICCIÓN DE ETIQUETAS (1: NORMAL, -1: ANOMALÍA)
        anomalies = labels == -1  # CONVERSIÓN A MÁSCARA BOOLEANA DE ANOMALÍAS
        logging.info(f"Anomalías detectadas por IF: {np.sum(anomalies)} de {len(anomalies)}.")  # LOG DE RESULTADOS DE DETECCIÓN
        return anomalies  # RETORNO DE MÁSCARA DE ANOMALÍAS

    def train_lstm_parallel(self, historical_inf, context_historical):
        # MÉTODO PARA ENTRENAR LA RED LSTM EN UN HILO PARALELO, EVITANDO BLOQUEO DEL FLUJO PRINCIPAL
        def _train():  # FUNCIÓN INTERNA QUE SE EJECUTARÁ EN EL HILO
            with self.training_lock:  # USO DE BLOQUEO PARA PREVENIR ACCESOS CONCURRENTES A RECURSOS COMPARTIDOS
                if len(historical_inf) <= self.seq_length:  # VERIFICACIÓN DE DATOS SUFICIENTES PARA SECUENCIAS
                    logging.warning("Datos insuficientes para LSTM.")
                    return
                
                full_data = np.hstack([historical_inf, context_historical])  # COMBINACIÓN HORIZONTAL DE DATOS INFRA Y CONTEXTO
                full_scaled = self.scaler_full.fit_transform(full_data)  # ESCALADO DE DATOS COMBINADOS
                
                X, y = self._prepare_sequences(full_scaled, self.seq_length)  # GENERACIÓN DE SECUENCIAS PARA ENTRENAMIENTO
                if len(X) == 0:  # VERIFICACIÓN DE SECUENCIAS VÁLIDAS GENERADAS
                    logging.warning("Secuencias insuficientes para LSTM.")
                    return
                
                # EXTRACCIÓN DE DIMENSIÓN DE SALIDA BASADA EN DATOS DE INFRAESTRUCTURA POST-PROCESADOS
                infra_dim = historical_inf.shape[1]
                y_inf = y[:, :infra_dim]  # OBJETIVOS SOLO PARA LA PARTE DE INFRAESTRUCTURA
                
                model = Sequential()  # CREACIÓN DE MODELO SECUENCIAL VACÍO
                model.add(Input(shape=(self.seq_length, full_scaled.shape[1])))  # CAPA DE ENTRADA DEFINIENDO FORMA DE SECUENCIAS
                model.add(LSTM(self.lstm_params['units']))  # CAPA LSTM CON NÚMERO DE UNIDADES ESPECIFICADO
                model.add(Dense(infra_dim))  # CAPA DE SALIDA DENSE CON DIMENSIÓN DE INFRA PARA REGRESIÓN
                model.compile(optimizer=Adam(), loss='mse')  # COMPILACIÓN CON OPTIMIZADOR ADAM Y PÉRDIDA MSE PARA REGRESIÓN
                
                # ENTRENAMIENTO DEL MODELO CON PARÁMETROS DE ÉPOCAS Y BATCH SIZE
                model.fit(X, y_inf, epochs=self.lstm_params['epochs'], batch_size=self.lstm_params['batch_size'], verbose=0)
                
                self.model_queue.put((model, self.scaler_full))  # DEPOSITA MODELO ENTRENADO Y SCALER EN COLA PARA RECUPERACIÓN
                logging.info("LSTM entrenado en paralelo.")  # LOG DE ENTRENAMIENTO COMPLETADO EN HILO
        
        thread = threading.Thread(target=_train)  # CREACIÓN DE HILO ASIGNANDO LA FUNCIÓN INTERNA
        thread.start()  # INICIO DEL HILO DE ENTRENAMIENTO
        logging.info("Hilo de entrenamiento LSTM iniciado.")  # LOG DE INICIO DEL PROCESO PARALELO
        return thread  # RETORNO DEL OBJETO HILO PARA MONITOREO OPCIONAL (EJ. JOIN)

    def update_lstm_model(self):
        # MÉTODO PARA ACTUALIZAR EL MODELO LSTM DESDE LA COLA, CON INTERVENCIÓN HUMANA (HITL) PARA APROBACIÓN
        if not self.model_queue.empty():  # VERIFICA SI HAY UN MODELO ENTRENADO DISPONIBLE EN LA COLA
            approve = input("[HITL] Aprobar nuevo modelo LSTM? (y/n): ")  # SOLICITUD DE APROBACIÓN MANUAL AL USUARIO
            if approve.lower() == 'y':  # SI SE APRUEBA, ACTUALIZA ATRIBUTOS
                self.lstm_model, scaler = self.model_queue.get()  # OBTIENE MODELO Y SCALER DE LA COLA
                self.scaler_full = scaler  # ACTUALIZA EL SCALER COMBINADO
                logging.info("Modelo LSTM actualizado (aprobado por HITL).")  # LOG DE ACTUALIZACIÓN EXITOSA
            else:
                logging.info("Modelo LSTM rechazado (ética HITL).")  # LOG DE RECHAZO POR RAZONES ÉTICAS O DE CALIDAD

    def run_lstm(self, realtime_inf, context_realtime):
        # MÉTODO PARA GENERAR PREDICCIONES CON LA LSTM EN DATOS EN TIEMPO REAL
        if self.lstm_model is None or len(realtime_inf) <= self.seq_length:  # VERIFICACIÓN DE MODELO Y DATOS SUFICIENTES
            logging.warning("LSTM no entrenado o datos insuficientes; retornando vacío.")
            return np.empty((0, realtime_inf.shape[1] if len(realtime_inf) > 0 else 0))
        
        full_realtime = np.hstack([realtime_inf, context_realtime])  # COMBINACIÓN DE DATOS INFRA Y CONTEXTO EN TIEMPO REAL
        full_scaled = self.scaler_full.transform(full_realtime)  # ESCALADO USANDO SCALER DEL ENTRENAMIENTO
        X, _ = self._prepare_sequences(full_scaled, self.seq_length)  # CREACIÓN DE SECUENCIAS PARA PREDICCIÓN
        if len(X) == 0:  # MANEJO DE SECUENCIAS INSUFICIENTES
            logging.warning("Secuencias insuficientes para predicción LSTM.")
            return np.empty((0, realtime_inf.shape[1]))
        
        preds_scaled = self.lstm_model.predict(X, verbose=0)  # GENERACIÓN DE PREDICCIONES EN ESCALA ESCALADA
        
        infra_dim = realtime_inf.shape[1]  # OBTENCIÓN DE DIMENSIÓN DE INFRA PARA DESESCALADO SELECTIVO
        dummy = np.zeros((len(preds_scaled), full_scaled.shape[1]))  # MATRIZ DUMMY PARA INVERSE TRANSFORM COMPLETO
        dummy[:, :infra_dim] = preds_scaled  # INSERCIÓN DE PREDICCIONES EN POSICIONES DE INFRA
        preds = self.scaler_full.inverse_transform(dummy)[:, :infra_dim]  # DESESCALADO Y EXTRACCIÓN DE PARTE RELEVANTE
        
        logging.info(f"Predicciones LSTM generadas: {len(preds)} samples.")  # LOG DE NÚMERO DE PREDICCIONES GENERADAS
        return preds  # RETORNO DE PREDICCIONES EN ESCALA ORIGINAL

    def diagnostic(self, anomalies, realtime_inf, preds):
        # MÉTODO PARA DIAGNOSTICAR ANOMALÍAS COMBINANDO RESULTADOS DE IF Y LSTM, IDENTIFICANDO FP, FN, ETC.
        diagnostics = []  # LISTA PARA ALMACENAR ETIQUETAS DE DIAGNÓSTICO POR MUESTRA
        fp_mask = np.zeros(len(realtime_inf), dtype=bool)  # MÁSCARA PARA FALSOS POSITIVOS DETECTADOS
        fn_mask = np.zeros(len(realtime_inf), dtype=bool)  # MÁSCARA PARA FALSOS NEGATIVOS
        confirmed_mask = np.zeros(len(realtime_inf), dtype=bool)  # MÁSCARA PARA ANOMALÍAS CONFIRMADAS
        
        offset = self.seq_length  # OFFSET PARA ALINEAR PREDICCIONES LSTM CON DATOS REALES (DADO EL DESPLAZAMIENTO DE SECUENCIAS)
        if len(realtime_inf) < offset or len(preds) == 0:  # VERIFICACIÓN DE DATOS SUFICIENTES PARA DIAGNÓSTICO
            logging.warning("Datos insuficientes para diagnóstico; retornando vacío.")
            return diagnostics, fp_mask, fn_mask, confirmed_mask
        
        logging.info("Iniciando diagnóstico de anomalías.")  # LOG DE INICIO DEL ANÁLISIS
        for i in range(offset, len(realtime_inf)):  # BUCLE SOBRE ÍNDICES DESDE OFFSET HASTA EL FINAL
            actual = realtime_inf[i]  # VALOR REAL DE INFRA EN POSICIÓN i
            pred = preds[i - offset]  # PREDICCIÓN LSTM CORRESPONDIENTE DESPLAZADA
            diff = mean_squared_error(actual, pred)  # CÁLCULO DE DIFERENCIA USANDO MSE
            is_anomaly_if = anomalies[i]  # BANDERA DE ANOMALÍA DESDE IF
            
            if is_anomaly_if:  # CASO: IF DETECTA ANOMALÍA
                if diff < self.diff_threshold:  # SI LSTM NO MUESTRA GRAN DIFERENCIA, ES FALSO POSITIVO
                    diagnostics.append('FP (falso positivo)')
                    fp_mask[i] = True
                else:  # SI LSTM CONFIRMA, ANOMALÍA REAL
                    diagnostics.append('Anomalía confirmada')
                    confirmed_mask[i] = True
            else:  # CASO: IF NO DETECTA
                if diff > self.diff_threshold:  # SI LSTM MUESTRA DIFERENCIA, FALSO NEGATIVO
                    diagnostics.append('FN (falso negativo)')
                    fn_mask[i] = True
                else:  # AMBOS COINCIDEN EN NORMALIDAD
                    diagnostics.append('Correcto')
        
        logging.info(f"Diagnósticos completados: {np.sum(fp_mask)} FP, {np.sum(fn_mask)} FN, {np.sum(confirmed_mask)} confirmadas.")  # LOG DE RESUMEN
        return diagnostics, fp_mask, fn_mask, confirmed_mask  # RETORNO DE RESULTADOS

    def supervision(self, preds, realtime_correct):
        # MÉTODO PARA SUPERVISAR EL RENDIMIENTO DE LSTM, CALCULAR ERROR Y ACTUALIZAR DATOS HISTÓRICOS
        if len(preds) == 0 or len(realtime_correct) == 0:  # VERIFICACIÓN DE DATOS DE ENTRADA VÁLIDOS
            logging.warning("Datos insuficientes para supervisión.")
            return
        
        min_len = min(len(realtime_correct), len(preds))  # DETERMINA LONGITUD MÍNIMA PARA COMPARACIÓN SEGURA
        if min_len == 0:  # MANEJO DE LONGITUD CERO
            logging.warning("No hay datos para calcular MSE en supervisión.")
            return
        
        error = mean_squared_error(realtime_correct[:min_len], preds[:min_len])  # CÁLCULO DE ERROR MSE EN DATOS COMUNES
        logging.info(f"Error de supervisión (MSE): {error:.4f}")  # LOG DEL ERROR CALCULADO
        
        if error > self.retrain_threshold:  # VERIFICACIÓN SI EL ERROR SUPERA UMBRAL, SUGIRIENDO REENTRENAMIENTO
            logging.warning("Umbral excedido: Recomendado reentrenar LSTM.")
        
        # ACTUALIZACIÓN DE DATOS HISTÓRICOS CONCATENANDO DATOS CORRECTOS ACUMULADOS
        self.historical_data = pd.concat([self.historical_data, self.correct_data])
        logging.info("Datos históricos actualizados con correctos.")  # LOG DE ACTUALIZACIÓN COMPLETADA