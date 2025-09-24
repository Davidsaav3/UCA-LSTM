import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(42)
# Datos normales: 1000 puntos en una distribución normal
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 2))
# Datos anómalos: 50 puntos alejados
anomalous_data = np.random.uniform(low=-6, high=6, size=(50, 2))
# Combinar datos
data = np.vstack([normal_data, anomalous_data])

# Crear y entrenar el modelo Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data)

# Predecir anomalías (-1 para anomalías, 1 para datos normales)
predictions = model.predict(data)

# Visualizar los resultados
plt.scatter(data[:, 0], data[:, 1], c=predictions, cmap='Paired')
plt.title('Detección de Anomalías con Isolation Forest')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Predicción (1: Normal, -1: Anómalo)')
plt.show()

# Mostrar resultados
df = pd.DataFrame(data, columns=['X', 'Y'])
df['Prediction'] = predictions
print("Primeras filas del conjunto de datos con predicciones:")
print(df.head())
print("\nNúmero de anomalías detectadas:", np.sum(predictions == -1))