import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# CARGA DE PARAMETROS
with open("parameters_config.json") as f:
    config = json.load(f)

params_if = config["isolation_forest_model"]
params_clustering = config.get("clustering", {"enabled": False})

if not params_clustering.get("enabled", False):
    print("Clustering deshabilitado en JSON. Saliendo...")
    exit()

df = pd.read_csv(params_if["output_path"])
anomalies = df[df["is_anomaly_detected"]==1].copy()

# ESCALADO
data = anomalies[[params_if["target_col"]]].values
if params_if.get("scaler")=="StandardScaler":
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
else:
    data_scaled = data

# CLUSTERING POR ALGORITMOS
for algo_params in params_clustering["algorithms"]:
    name = algo_params["name"]
    print(f"Ejecutando clustering: {name}")

    if name=="KMeans":
        model = KMeans(n_clusters=algo_params.get("n_clusters",5), random_state=algo_params.get("random_state",42))
        labels = model.fit_predict(data_scaled)
    elif name=="DBSCAN":
        model = DBSCAN(eps=algo_params.get("eps",0.5), min_samples=algo_params.get("min_samples",5))
        labels = model.fit_predict(data_scaled)
    elif name=="Agglomerative":
        model = AgglomerativeClustering(n_clusters=algo_params.get("n_clusters",5), linkage=algo_params.get("linkage","ward"))
        labels = model.fit_predict(data_scaled)
    else:
        raise NotImplementedError(f"Algoritmo {name} no implementado")

    anomalies.loc[:, f"cluster_{name}"] = labels

    # GUARDAR CSV
    os.makedirs(os.path.dirname(algo_params["output_csv"]), exist_ok=True)
    anomalies.to_csv(algo_params["output_csv"], index=False)

    # GRAFICO
    plt.figure(figsize=(8,5))
    for cluster_id in np.unique(labels):
        cluster_data = anomalies[anomalies[f"cluster_{name}"]==cluster_id]
        plt.scatter(cluster_data.index, cluster_data[params_if["target_col"]],
                    label=f"Cluster {cluster_id}", alpha=0.7)
    plt.xlabel("Índice")
    plt.ylabel(params_if["target_col"])
    plt.title(f"Clustering de anomalías ({name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(algo_params["output_plot"])
    plt.close()

    # METRICA DE CALIDAD
    try:
        sil_score = silhouette_score(data_scaled, labels)
        print(f"Silhouette score {name}: {sil_score:.4f}")
    except:
        print(f"Silhouette score no calculable para {name}")

print("Clustering completado ✅")
