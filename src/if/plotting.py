import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize

def plot_anomalies(df_original, results_df, features_to_plot, timestamp, plot_path, show_plots=True):
    plot_features = df_original.columns if features_to_plot is None else features_to_plot
    for feature in plot_features:
        plt.figure(figsize=(12,6))
        plt.plot(df_original[feature], label="Valor Real")
        norm = Normalize(vmin=0, vmax=1)
        plt.scatter(
            results_df.index[results_df["Anomalia"]==1],
            df_original.loc[results_df["Anomalia"]==1, feature],
            color=plt.cm.Reds(norm(results_df.loc[results_df["Anomalia"]==1,"Score_norm"])),
            label="Anomalía",
            s=50
        )
        plt.title(f"Detección de anomalías - {feature}", fontsize=14)
        plt.xlabel("Índice de muestra")
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs(plot_path, exist_ok=True)
        plot_file = f"{plot_path}/anomaly_{feature}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300)
        print(f"✅ Gráfico guardado: {plot_file}")
        if show_plots:
            plt.show()
        else:
            plt.close()
