let mobilenetModel;
let classifier;
let xs = [];
let ys = [];
let labels = [];
let labelToIndex = {};

const statusEl = document.getElementById("status");
const trainingImagesEl = document.getElementById("trainingImages");
const testImagePreviewEl = document.getElementById("testImagePreview");

async function loadMobileNet() {
  mobilenetModel = await mobilenet.load();
  statusEl.innerText = "MobileNet loaded.";
}

function addExample() {
  const fileInput = document.getElementById("imageInput");
  const labelInput = document.getElementById("labelInput");
  const file = fileInput.files[0];
  const label = labelInput.value.trim();

  if (!file || !label) {
    statusEl.innerText = "Please select an image and enter a label.";
    return;
  }

  const reader = new FileReader();
  reader.onload = async () => {
    const img = new Image();
    img.src = reader.result;
    img.onload = async () => {
      // Display the training image with label
      const imgElement = document.createElement("img");
      imgElement.src = reader.result;
      
      const labelDiv = document.createElement("div");
      labelDiv.innerHTML = `<strong>${label}</strong>`;
      labelDiv.appendChild(imgElement);
      trainingImagesEl.appendChild(labelDiv);

      // Prepare the image for training
      const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
      const embedding = mobilenetModel.infer(tensor, 'conv_preds');

      xs.push(embedding);

      if (!(label in labelToIndex)) {
        labelToIndex[label] = Object.keys(labelToIndex).length;
        labels.push(label);
      }

      ys.push(labelToIndex[label]);
      statusEl.innerText = `Added example for "${label}"`;
    };
  };
  reader.readAsDataURL(file);
}

async function trainModel() {
  if (xs.length === 0) {
    statusEl.innerText = "Add examples before training.";
    return;
  }

  const xTrain = tf.concat(xs);
  const yTrain = tf.tensor1d(ys, 'int32');
  const yOneHot = tf.oneHot(yTrain, labels.length);

  classifier = tf.sequential();
  classifier.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
  classifier.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));
  classifier.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  statusEl.innerText = "Training model...";

  await classifier.fit(xTrain, yOneHot, {
    epochs: 20,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        statusEl.innerText = `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} acc=${logs.acc.toFixed(4)}`;
      },
      onTrainEnd: () => {
        statusEl.innerText = "✅ Model trained successfully!";
      }
    }
  });
}

function predictImage() {
  const fileInput = document.getElementById("testInput");
  const file = fileInput.files[0];
  if (!file || !classifier) {
    statusEl.innerText = "Please train the model and select an image to predict.";
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.src = reader.result;
    img.onload = async () => {
      // Display the test image
      testImagePreviewEl.innerHTML = '';
      const imgElement = document.createElement("img");
      imgElement.src = reader.result;
      testImagePreviewEl.appendChild(imgElement);

      const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
      const embedding = mobilenetModel.infer(tensor, 'conv_preds');
      const prediction = classifier.predict(embedding);

      // Get the probabilities of the prediction
      const probabilities = prediction.dataSync();
      const maxProb = Math.max(...probabilities);
      const predictedIndex = probabilities.indexOf(maxProb);

      // Check if the max probability is above a certain threshold (e.g., 0.7)
      const confidenceThreshold = 0.7;

      if (maxProb > confidenceThreshold && predictedIndex < labels.length) {
        document.getElementById("prediction").innerText = `🧠 Predicted: ${labels[predictedIndex]}`;
      } else {
        document.getElementById("prediction").innerText = "⚠️ Not predicted, model has not been trained on this label.";
      }
    };
  };
  reader.readAsDataURL(file);
}

// Toggle visibility of training images
function toggleImages() {
  const imagesDiv = document.getElementById("trainingImages");
  const button = document.getElementById("showImagesButton");

  if (imagesDiv.style.display === "none") {
    imagesDiv.style.display = "block";
    button.innerText = "Hide Training Images";
  } else {
    imagesDiv.style.display = "none";
    button.innerText = "Show Training Images";
  }
}

loadMobileNet();
