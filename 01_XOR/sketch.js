let data;

let predictionMap;
let resolution = 50;

function setup() {
  createCanvas(400, 400);
  nn = new NeuralNetwork(2, 4, 2);
  data = {
    inputs: tf.tensor2d([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ]),
    targets: tf.tensor2d([
      [0, 0],
      [1, 1],
      [1, 1],
      [0, 0]
    ])
  };
  colorMode(RGB,1);
  let scratch = [];
  let cols = width / resolution;
  let rows = height / resolution;
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      let x1 = i / cols;
      let x2 = j / rows;
      scratch.push([x1, x2]);
    }
  }
  predictionMap = tf.tensor2d(scratch);
}

function draw() {
  let predictions = nn.predict(predictionMap);
  // nn.train(data);
  let cols = width / resolution;
  let rows = height / resolution;
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      let y = predictions[i*2+j*cols*2];
      noStroke();
      fill(y);
      rect(i * resolution, j * resolution, resolution, resolution);
    }
  }

}