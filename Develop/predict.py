from argparse import ArgumentParser

import yaml
from sklearn.metrics import classification_report

from shotTypeML_pkg.imageUtil import *
from shotTypeML_pkg.loadModel import *

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

shotTypes = ['CU', 'MS', 'LS', 'ELS']


def predictShotType(modelPath, modelWeightsPath, testDataPath, targetImageSize):
    # Load test frames path from config
    testFramesPath = testDataPath

    # Load model and weights
    if os.path.exists(modelPath):
        # Load model
        print("load model from " + modelPath)
        model = keras.models.load_model(modelPath)
        if os.path.exists(modelWeightsPath):
            # Load weights
            print("load weights from " + modelWeightsPath)
            model.load_weights(modelWeightsPath)

    # Load test data
    print("loading test data...")
    testFrames, testLabels = loadImagesAndLabels(testFramesPath, shotTypes, targetImageSize, True)

    # Predict test data shot types
    results = model.predict(testFrames, verbose=1)
    results_bool = np.argmax(results, axis=1)
    print(classification_report(testLabels, results_bool))


if __name__ == '__main__':
    print()
    parser = ArgumentParser()
    parser.add_argument('-config', type=str, help='Config .yaml file containing configuration settings', required=True)
    args = parser.parse_args()

    with open(args.config) as configFile:
        config = yaml.full_load(configFile)

    predictShotType(
        config['model'],
        config['modelWeights'],
        config['testFrames'],
        int(config['targetImageSize'])
    )
