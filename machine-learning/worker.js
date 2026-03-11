importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_IMAGE_DIMENSION = 640;
const CLASS_THRESHOLD = 0.7;

let _model = null;
let _labels = [];

async function loadModel() {
    const model = await tf.loadGraphModel(MODEL_PATH);
    return model;
}

async function loadLabels() {
    const labels = await (await fetch(LABELS_PATH)).json();
    return labels;
}

async function startModel() {

    console.log('🧠 YOLOv5n Web Worker starting...');
    await tf.ready();

    _model = await loadModel();
    _labels = await loadLabels();

    const dummyInput = tf.ones(_model.inputs[0].shape); // Warm-up the model
    await _model.executeAsync(dummyInput);
    tf.dispose(dummyInput);

    postMessage({
        type: 'model-loaded'
    });

}

function preProcessInput(input) {

    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input);
        return tf.image
            .resizeBilinear(image, [INPUT_IMAGE_DIMENSION, INPUT_IMAGE_DIMENSION])
            .div(255)
            .expandDims(0);
    });
}

function* postProcessOutput({ boxes, scores, classes }, width, height) {
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] < CLASS_THRESHOLD) continue;
        const label = _labels[classes[i]];

        if (label != 'kite') continue;

        let [x1, y1, x2, y2] = boxes.slice(i * 4, i * 4 + 4);
        x1 *= width;
        y1 *= height;
        x2 *= width;
        y2 *= height;

        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;

        const centerX = x1 + boxWidth / 2;
        const centerY = y1 + boxHeight / 2;

        const score = (scores[i] * 100).toFixed(2);

        yield {
            x: centerX,
            y: centerY,
            score,
            label
        };
    }
}

async function runInference(tensor) {
    const output = await _model.executeAsync(tensor);
    tf.dispose(tensor);
    const [boxes, scores, classes] = output.slice(0, 3);
    const [boxesData, scoresData, classesData] = await Promise.all([
        boxes.data(),
        scores.data(),
        classes.data()
    ]);
    output.forEach(t => t.dispose());

    return { boxes: boxesData, scores: scoresData, classes: classesData };
}


self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return
    if (!_model || !_labels) return;

    const input = preProcessInput(data.image);
    const { width, height } = data.image;
    const inferenceResult = await runInference(input);

    for (const detection of postProcessOutput(inferenceResult, width, height)) {
        postMessage({
            type: 'prediction',
            ...detection
        });
    }




};

startModel();

console.log('🧠 YOLOv5n Web Worker initialized');
