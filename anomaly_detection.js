import * as tf from '@tensorflow/tfjs';

// Clase para simular Isolation Forest
class IsolationForestSimulator {
  constructor(numTrees = 100, maxDepth = 10) {
    this.numTrees = numTrees;
    this.maxDepth = maxDepth;
    this.trees = [];
  }

  // Generar un árbol de decisión simple para simular IF
  buildTree(data, depth = 0) {
    if (depth >= this.maxDepth || data.length <= 1) return null;
    
    const feature = Math.floor(Math.random() * data[0].length);
    const values = data.map(d => d[feature]);
    const splitValue = values[Math.floor(Math.random() * values.length)];
    
    const left = data.filter(d => d[feature] <= splitValue);
    const right = data.filter(d => d[feature] > splitValue);
    
    return {
      feature,
      splitValue,
      left: this.buildTree(left, depth + 1),
      right: this.buildTree(right, depth + 1)
    };
  }

  // Entrenar el modelo
  train(data) {
    for (let i = 0; i < this.numTrees; i++) {
      this.trees.push(this.buildTree(data));
    }
  }

  // Calcular puntuación de anomalía
  score(sample) {
    let totalPathLength = 0;
    for (const tree of this.trees) {
      let pathLength = 0;
      let node = tree;
      while (node && pathLength < this.maxDepth) {
        if (!node.left && !node.right) break;
        pathLength++;
        node = sample[node.feature] <= node.splitValue ? node.left : node.right;
      }
      totalPathLength += pathLength;
    }
    return totalPathLength / this.numTrees;
  }

  // Calcular indicadores contextuales (IR e II)
  computeContextualIndicators(data, contexts) {
    const ir = new Set(data.map(d => contexts[d.context])).size;
    const ii = data.reduce((sum, d) => sum + Math.abs(d.value), 0) / data.length;
    return { ir, ii };
  }
}

// Clase para manejar la red LSTM por contexto
class ContextualLSTM {
  constructor(context, inputShape, units = 64) {
    this.context = context;
    this.model = this.buildModel(inputShape, units);
  }

  buildModel(inputShape, units) {
    const model = tf.sequential();
    model.add(tf.layers.lstm({
      units,
      inputShape,
      returnSequences: false
    }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({
      optimizer: 'adam',
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
    return model;
  }

  async train(X, y, epochs = 50, batchSize = 32) {
    const xs = tf.tensor3d(X);
    const ys = tf.tensor2d(y);
    await this.model.fit(xs, ys, {
      epochs,
      batchSize,
      shuffle: true
    });
    xs.dispose();
    ys.dispose();
  }

  async predict(data) {
    const input = tf.tensor3d([data]);
    const prediction = await this.model.predict(input).data();
    input.dispose();
    return prediction[0];
  }
}

// Clase principal para el sistema híbrido
class HybridAnomalyDetector {
  constructor(contexts) {
    this.ifSimulator = new IsolationForestSimulator();
    this.lstms = {};
    this.contexts = contexts;
    contexts.forEach(context => {
      this.lstms[context] = new ContextualLSTM(context, [10, 5]); // Ejemplo: 10 pasos temporales, 5 características
    });
  }

  // Preparar datos para IF y LSTM
  preprocessData(rawData) {
    return rawData.map(d => ({
      context: d.context,
      value: d.value,
      features: d.features // Características numéricas para IF
    }));
  }

  async train(data) {
    const preprocessedData = this.preprocessData(data);
    
    // Entrenar Isolation Forest
    this.ifSimulator.train(preprocessedData.map(d => d.features));
    
    // Entrenar LSTMs por contexto
    for (const context of this.contexts) {
      const contextData = preprocessedData.filter(d => d.context === context);
      const X = contextData.map(d => Array(10).fill(d.features)); // Secuencia temporal simulada
      const y = contextData.map(d => d.isAnomaly ? 1 : 0); // Etiquetas de anomalía
      await this.lstms[context].train(X, y);
    }
  }

  async detect(dataPoint) {
    const features = dataPoint.features;
    const context = dataPoint.context;

    // Paso 1: Detección inicial con IF
    const ifScore = this.ifSimulator.score(features);
    const contextualIndicators = this.ifSimulator.computeContextualIndicators([dataPoint], this.contexts);

    // Paso 2: Refinamiento con LSTM
    const lstmScores = {};
    for (const ctx of this.contexts) {
      const input = Array(10).fill(features); // Secuencia temporal simulada
      lstmScores[ctx] = await this.lstms[ctx].predict(input);
    }

    // Paso 3: Enriquecimiento de la anomalía
    return {
      ifScore,
      recurrenceIndicator: contextualIndicators.ir,
      impactIndicator: contextualIndicators.ii,
      lstmScores,
      metadata: {
        timestamp: new Date().toISOString(),
        context,
        additionalAttributes: dataPoint.metadata || {}
      }
    };
  }

  // Explicabilidad: Identificar contribución de cada subred
  explain(result) {
    const contributions = Object.entries(result.lstmScores)
      .map(([context, score]) => ({ context, score }))
      .sort((a, b) => b.score - a.score);
    
    return {
      primaryContributor: contributions[0].context,
      contributionScores: contributions,
      ifScore: result.ifScore,
      summary: `La anomalía está más correlacionada con el contexto "${contributions[0].context}" (puntuación: ${contributions[0].score.toFixed(2)})`
    };
  }
}

// Ejemplo de uso
async function main() {
  const contexts = ['festivos', 'climatologica', 'operativa', 'demografica'];
  const detector = new HybridAnomalyDetector(contexts);

  // Datos de ejemplo
  const sampleData = [
    { context: 'festivos', value: 100, features: [1, 2, 3, 4, 5], isAnomaly: 1, metadata: { day: 'monday' } },
    { context: 'climatologica', value: 50, features: [2, 3, 4, 5, 6], isAnomaly: 0, metadata: { temp: 25 } },
    // Más datos...
  ];

  // Entrenar el modelo
  await detector.train(sampleData);

  // Detectar anomalía en un nuevo punto
  const newDataPoint = {
    context: 'festivos',
    features: [1, 2, 3, 4, 5],
    metadata: { day: 'sunday' }
  };
  const result = await detector.detect(newDataPoint);

  // Explicar resultado
  const explanation = detector.explain(result);
  console.log('Resultado de detección:', result);
  console.log('Explicación:', explanation);
}

main().catch(console.error);