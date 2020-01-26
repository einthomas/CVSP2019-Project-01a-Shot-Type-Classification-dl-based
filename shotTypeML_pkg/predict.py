from shotTypeML_pkg.imageUtil import *
from shotTypeML_pkg.loadModel import *

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

shotTypes = ['CU', 'MS', 'LS', 'ELS']
model = 0


def predictShotType(modelPath, modelWeightsPath, image):
    """ Predicts the shot type of the provided image. """

    # Load model and weights if they have not been loaded
    global model
    if model == 0:
        # Load model and weights
        if os.path.exists(modelPath):
            # Load model
            print("load model from " + modelPath)
            model = keras.models.load_model(modelPath)
            if os.path.exists(modelWeightsPath):
                # Load weights
                print("load weights from " + modelWeightsPath)
                model.load_weights(modelWeightsPath)

    # Predict image data shot types
    predictions = model.predict(image)
    labels = [shotTypes[np.argmax(prediction)] for prediction in predictions]
    return labels
