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
    if os.path.exists(getConfigRelativePath('model')):
        # Load model
        print("load model from " + getConfigRelativePath('model'))
        model = keras.models.load_model(getConfigRelativePath('model'))
        if os.path.exists(getConfigRelativePath('modelWeights')):
            # Load weights
            print("load weights from " + getConfigRelativePath('modelWeights'))
            model.load_weights(getConfigRelativePath('modelWeights'))

    targetImageSize = 224

    # Load test data
    print("loading test data...")
    testFrames, testLabels = loadImagesAndLabels(testFramesPath, shotTypes, targetImageSize, True)
    testLabelsCategorical = to_categorical(testLabels)

    # Predict test data shot types
    results = model.predict(testFrames, verbose=1)
    results_bool = np.argmax(results, axis=1)
    print(classification_report(testLabels, results_bool))


model = 0
def predictShotType_production(images):
    global model
    if model == 0:
        # Load model and weights
        if os.path.exists(getConfigRelativePath('model')):
            # Load model
            print("load model from " + getConfigRelativePath('model'))
            model = keras.models.load_model(getConfigRelativePath('model'))
            if os.path.exists(getConfigRelativePath('modelWeights')):
                # Load weights
                print("load weights from " + getConfigRelativePath('modelWeights'))
                model.load_weights(getConfigRelativePath('modelWeights'))

    # Predict image data shot types
    predictions = model.predict(images)
    labels = [shotTypes[np.argmax(prediction)] for prediction in predictions]
    return labels


if __name__ == '__main__':
    predictShotType_testData()
