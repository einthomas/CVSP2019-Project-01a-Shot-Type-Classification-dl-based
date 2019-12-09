import datetime
import os

import numpy as np
import tensorflow as tf
import yaml
from keras.preprocessing import image
from keras.utils import to_categorical
from tensorflow import keras

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

executablePath = os.path.dirname(os.path.realpath(__file__))

# Load config.yaml
config = {}
with open(os.path.join(executablePath, 'config.yaml')) as configFile:
    config = yaml.full_load(configFile)


######### MODEL LOADING #########

# Load pretrained model for transfer learning
oldModel = keras.models.load_model(config['modelPath'])

# Discard the last two layers (global avg pooling and the last dense layer)
layers = oldModel.layers[len(oldModel.layers) - 3].output

# Add two new layers
layers = keras.layers.GlobalAveragePooling2D()(layers)
layers = keras.layers.Dense(4, activation='softmax')(layers)

# Replace the input layer to change the input shape
oldModel._layers[0]._batch_input_shape = (None, 224, 224, 3)

# Build new model
model = keras.models.Model(inputs=oldModel.input, outputs=layers)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    # optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

with open(os.path.join(executablePath, 'Logs\\modelSummary.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))


######### LOAD DATA #########

# Resize an image and apply a center crop. The returned image is a square with
# a width and height of targetSize
def centerCropImage(img, targetSize):
    # Resize image while keeping its aspect ratio
    width, height = img.size
    aspectRatio = width / height
    resizedWidth = int(targetSize * aspectRatio)
    resizedWidth = resizedWidth + resizedWidth % 2
    img = img.resize((resizedWidth, targetSize))

    # Apply a center crop by cutting away the same number of pixels at both
    # sides, left and right, of the image
    width, height = img.size
    offsetX = round((width - targetSize) * 0.5)
    return img.crop((offsetX, 0, width - offsetX, height))


# Loads the frames located in path. It is assumed that the frames are located
# in folders named according to their shot type (CU, MS, LS or ELS)
def loadFramesLabels(path, shotTypes, targetSize):
    frames = []
    labels = []

    for shotType in shotTypes:
        currentPath = os.path.join(path, shotType)
        for imageName in os.listdir(currentPath):
            labels.append(shotTypes.index(shotType))

            # Load, scale and crop image
            img = image.load_img(os.path.join(currentPath, imageName), color_mode="grayscale")
            img = centerCropImage(img, targetSize)
            img = image.img_to_array(img)
            img = img / 255.0

            # reshape image from (224, 224, 1) to (224, 224, 3)
            img = np.squeeze(np.stack((img,)*3, axis=-1))

            frames.append(img)

    return np.array(frames), np.array(labels)


shotTypes = ['CU', 'MS', 'LS', 'ELS']
targetSize = 224

# Load training data
trainFrames, trainLabels = loadFramesLabels(config['trainFramesPath'], shotTypes, targetSize)
trainLabels = to_categorical(trainLabels)

# Load validation data
valFrames, valLabels = loadFramesLabels(config['valFramesPath'], shotTypes, targetSize)
valLabels = to_categorical(valLabels)


######### MODEL TRAINING #########
log_dir = os.path.join(executablePath, "Logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    trainFrames,
    trainLabels,
    validation_data=(valFrames, valLabels),
    callbacks=[tensorboard_callback],
    epochs=10
)

model.save_weights(config['modelWeightsPath'])
