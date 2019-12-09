import os
import tensorflow as tf
from Common.loadModel import loadModel

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


######### MODEL LOADING #########
model = loadModel()


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
