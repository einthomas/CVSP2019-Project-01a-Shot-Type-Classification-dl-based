from keras.utils import to_categorical
from sklearn.metrics import classification_report

from Common.imageUtil import *
from Common.loadModel import *

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

shotTypes = ['CU', 'MS', 'LS', 'ELS']


def predictShotType_testData():
    # Load test frames path from config
    testFramesPath = config['testFrames']

    # Load model and weights
    if os.path.exists(getConfigRelativePath('modifiedModel')):
        # Load model
        print("load model from " + getConfigRelativePath('modifiedModel'))
        model = keras.models.load_model(getConfigRelativePath('modifiedModel'))
        if os.path.exists(getConfigRelativePath('checkpointModel')):
            # Load weights
            print("load weights from " + getConfigRelativePath('checkpointModel'))
            model.load_weights(getConfigRelativePath('checkpointModel'))

    targetImageSize = 224

    # Load test data
    print("loading test data...")
    testFrames, testLabels = loadImagesAndLabels(testFramesPath, shotTypes, targetImageSize, True)
    testLabelsCategorical = to_categorical(testLabels)

    # Predict test data shot types
    results = model.predict(testFrames, verbose=1)
    results_bool = np.argmax(results, axis=1)
    print(classification_report(testLabels, results_bool))


def predictShotType_production(images):
    # Load model and weights
    if os.path.exists(getConfigRelativePath('modifiedModel')):
        # Load model
        print("load model from " + getConfigRelativePath('modifiedModel'))
        model = keras.models.load_model(getConfigRelativePath('modifiedModel'))
        if os.path.exists(getConfigRelativePath('checkpointModel')):
            # Load weights
            print("load weights from " + getConfigRelativePath('checkpointModel'))
            model.load_weights(getConfigRelativePath('checkpointModel'))

    # Predict image data shot types
    predictions = model.predict(images)
    labelPredictions = [shotTypes[np.argmax(prediction)] for prediction in predictions]
    return labelPredictions


if __name__ == '__main__':
    predictShotType_testData()
