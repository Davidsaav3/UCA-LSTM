from sklearn.ensemble import IsolationForest
import time

def train_iforest(scaled_data, n_estimators, max_samples, max_features, contamination, bootstrap=True, random_state=42):
    start_time = time.time()
    print("🌲 Entrenando Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        verbose=0
    )
    iso_forest.fit(scaled_data)
    preds = iso_forest.predict(scaled_data)
    scores = iso_forest.decision_function(scaled_data)
    end_time = time.time()
    print(f"⏱️ Entrenamiento completado en {end_time - start_time:.2f} s")
    return iso_forest, preds, scores, end_time - start_time
