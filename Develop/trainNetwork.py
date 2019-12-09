import datetime

import tensorflow as tf
from keras.utils import to_categorical

from Common.imageUtil import *
from Common.loadModel import *
from Common.util import *

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


######### LOAD MODEL #########
model = loadModel()


######### LOAD DATA #########
shotTypes = ['CU', 'MS', 'LS', 'ELS']
targetSize = 224

# Load training data
trainFrames, trainLabels = loadFramesLabels(config['trainFramesPath'], shotTypes, targetSize)
trainLabels = to_categorical(trainLabels)

# Load validation data
valFrames, valLabels = loadFramesLabels(config['valFramesPath'], shotTypes, targetSize)
valLabels = to_categorical(valLabels)


######### TRAIN MODEL #########
logDir = os.path.join(getConfigRelativePath('developTrainingLogs') + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=1)

history = model.fit(
    trainFrames,
    trainLabels,
    validation_data=(valFrames, valLabels),
    callbacks=[tensorboardCallback],
    epochs=10
)

model.save_weights(config['modelWeightsPath'])
