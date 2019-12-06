import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.utils import to_categorical
import os
import csv
import numpy as np
import cv2

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


######### MODEL LOADING #########

# Load pretrained model for transfer learning
modelPath = 'D:\\CSVP2019\\model\\model_shotscale_967.h5'
oldModel = keras.models.load_model(modelPath)

# Discard the last two layers (global avg pooling and the last dense layer)
layers = oldModel.layers[len(oldModel.layers) - 3].output

# Add two new layers
layers = keras.layers.GlobalAveragePooling2D()(layers)
layers = keras.layers.Dense(4, activation='softmax')(layers)

# Build new model
model = keras.models.Model(inputs=oldModel.input, outputs=layers)
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    #optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='D:\\CSVP2019\\model\\model_transfer.h5',
    monitor='val_acc',
    save_best_only=True
)


######### LOAD DATA #########

# Resize an image and apply a center crop. The returned image is a square with
# a width and height of targetSize
def centerCropImage(img, targetSize):
    # Resize image while keeping its aspect ratio
    width, height = img.size
    aspectRatio = width / height
    img = img.resize((int(targetSize * aspectRatio), targetSize))
    
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
        currentPath = os.path.join(path, shotTypes)
        for imageName in os.listdir(currentPath):
            labels.append(shotTypes)

            # Load, scale and crop image
            img = image.load_img(os.path.join(currentPath, imageName + '.jpg'), grayscale=True)
            img = centerCropImage(img, targetSize)
            img = image.img_to_array(img)
            img = img / 255.0
            frames.append(img)
    
    return np.array(frames), np.array(labels)

shotTypes = ['CU', 'MS', 'LS', 'ELS']
targetSize = 224

# Load training data
trainFramesPath = 'D:\\CSVP2019\\copied_merge_new\\train'
trainFrames, trainLabels = loadFramesLabels(trainFramesPath, shotTypes, targetSize)
trainLabels = to_categorical(trainLabels)

# Load validation data
valFramesPath = 'D:\\CSVP2019\\copied_merge_new\\val'
valFrames, valLabels = loadFramesLabels(valFramesPath, shotTypes, targetSize)
valLabels = to_categorical(valLabels)


######### MODEL TRAINING #########

history = model.fit(
    trainFrames,
    trainLabels,
    #validation_data=(valFrames, valLabels),
    #callbacks=[checkpoint],
    epochs=10,
    validation_split=0.1
)

#history_df = pd.DataFrame(history.history)
#history_df[['loss', 'val_loss']].plot()
#history_df[['acc', 'val_acc']].plot()

#a = model.predict(valFrames)
#a.to_csv('D:\\CSVP2019\\copied_merge\\asdf.csv')

model.save_weights('D:\\CSVP2019\\model\\trained_model_weights.h5')
