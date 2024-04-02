async function loadTFModel() {
    const TFJS_BACKEND = 'webgl'
    const MODEL_PATH = 'tfjs_model/model.json';

    await tf.setBackend(TFJS_BACKEND);
    await tf.ready();
    return await tf.loadGraphModel(MODEL_PATH);
}

async function runTFModel() {
    const model = await loadTFModel();

    const imageEmbeddingsShape = [1, 256, 64, 64]
    const imageEmbeddings = tf.randomUniform(imageEmbeddingsShape, 0, 1);

    const pointLabelsShape = [1, 1, 3]
    const pointLabels = tf.randomUniform(pointLabelsShape, 0, 1);

    const origImgSize = tf.tensor([1024, 1024], [2], 'int32')

    const pointCoordsShape = [1, 1, 3, 2]
    const pointCoords = tf.randomUniform(pointCoordsShape, 0, 1);

    console.log(`imageEmbeddings.shape : ${imageEmbeddings.shape}`);
    console.log(`pointLabels.shape : ${pointLabels.shape}`);
    console.log(`origImgSize.shape : ${origImgSize.shape}`);
    console.log(`pointCoords.shape : ${pointCoords.shape}`);

    const inputs = {
        'image_embeddings': imageEmbeddings,
        'batched_point_labels': pointLabels,
        'orig_im_size': origImgSize,
        'batched_point_coords': pointCoords
    }
    const outputs = await model.executeAsync(inputs);
}

runTFModel();