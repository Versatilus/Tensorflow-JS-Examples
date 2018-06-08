// Based on: https://github.com/tensorflow/tfjs-examples/tree/master/mnist

class NeuralNetwork {

  constructor(inputs, hidden, outputs) {
    this.isTraining = false;
    this.model = tf.sequential();
    const hiddenLayer = tf.layers.dense({
      units: hidden,
      inputShape: [inputs],
      activation: 'sigmoid'
    });
    const outputLayer = tf.layers.dense({
      units: outputs,
      activation: 'sigmoid'
    });
    this.model.add(hiddenLayer);
    this.model.add(outputLayer);
    this.model.compile({
      optimizer: tf.train.sgd(0.1), // The default learning rate is too slow
      loss: 'meanSquaredError'
    });
  }

  predict(inputs) {
    return tf.tidy(()=>{
      const xs = Array.isArray(inputs)?Array.isArray(inputs[0])? tf.tensor(inputs):tf.tensor([inputs]):inputs;
      return this.model.predict(xs).dataSync();
    });
  }

  async train(data) {
    const isTensor = x=>x instanceof tf.Tensor;
    const xs = isTensor(data.inputs) ? data.inputs: tf.tensor2d(data.inputs);
    const ys = isTensor(data.targets) ? data.targets: tf.tensor2d(data.targets);
    const history = await this.model.fit(xs, ys, {
      epochs: 10,
      shuffle: true,
      validationData: [xs, ys]
    });
    if (!isTensor(data.inputs)) xs.dispose();
    if (!isTensor(data.targets)) ys.dispose();
    return history;
  }
}