from keras.utils import to_categorical
from sklearn.metrics import classification_report

from Common.imageUtil import *
from Common.loadModel import *

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

######### LOAD CONFIG #########
testFramesPath = config['testFrames']

######### LOAD MODEL #########
if os.path.exists(getConfigRelativePath('modifiedModel')):
    # Load model
    print("load model from " + getConfigRelativePath('modifiedModel'))
    model = keras.models.load_model(getConfigRelativePath('modifiedModel'))
    if os.path.exists(getConfigRelativePath('checkpointModel')):
        # Load weights
        print("load weights from " + getConfigRelativePath('checkpointModel'))
        model.load_weights(getConfigRelativePath('checkpointModel'))

######### LOAD DATA #########
shotTypes = ['CU', 'MS', 'LS', 'ELS']
targetImageSize = 224

# Load test data
print("loading test data...")
testFrames, testLabels = loadFramesLabels(testFramesPath, shotTypes, targetImageSize, True)
testLabelsCategorical = to_categorical(testLabels)

######### PREDICT #########
results = model.predict(testFrames, verbose=1)
results_bool = np.argmax(results, axis=1)
print(classification_report(testLabels, results_bool))
