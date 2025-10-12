def auto_adjust_params(scaled_data):
    n_samples, n_features = scaled_data.shape
    n_estimators = min(max(int(n_samples / 500), 50), 500)
    max_samples = min(max(0.1, n_samples / 10000), 1.0)
    max_features = min(max(0.5, n_features / 50), 1.0)
    contamination = 0.05
    print(f"✅ Parámetros automáticos ajustados: n_estimators={n_estimators}, max_samples={max_samples:.2f}, max_features={max_features:.2f}, contamination={contamination:.2f}")
    return n_estimators, max_samples, max_features, contamination
