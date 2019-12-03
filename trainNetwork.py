import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.utils import to_categorical
import os
import csv
import numpy as np
import pandas as pd

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


######### MODEL LOADING #########

# Load pretrained model for transfer learning
oldModel = keras.models.load_model('D:\\CSVP2019\\model\\model_shotscale_967.h5')

# Discard the last two layers (global avg pooling and the last dense layer)
layers = oldModel.layers[len(oldModel.layers) - 3].output

# Add two new layers
layers = keras.layers.GlobalAveragePooling2D()(layers)
layers = keras.layers.Dense(4, activation='softmax')(layers)

# Build new model
model = keras.models.Model(inputs=oldModel.input, outputs=layers)
model.compile(
    #optimizer=keras.optimizers.Adam(lr=0.00001),
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='D:\\CSVP2019\\model\\model_transfer.h5',
    save_best_only=True
)


######### LOAD DATA #########

shotTypes = ['CU', 'MS', 'LS', 'ELS']
targetSize = (125, 224, 3)

# Load frames and labels from CSV
def parseAnnotationsCSV(csvPath, framesPath, targetSize):
    frames = []
    labels = []
    
    csvReader = csv.reader(open(csvPath), delimiter=';')
    for row in csvReader:
        # Skip header
        if csvReader.line_num == 1:
            continue

        frameName = row[1]
        frameNumber = row[2]
        label = row[3]
        
        if label != 'NONE':
            labels.append(shotTypes.index(label))

            # Load and scale image
            img = image.load_img(os.path.join(framesPath, frameName + '.jpg'), target_size=targetSize)
            img = image.img_to_array(img)
            img = img / 255.0
            frames.append(img)
    
    return np.array(frames), np.array(labels)

trainFramesPath = 'D:\\CSVP2019\\copied\\data\\test\\imc'
trainAnnotationsCSVPath = 'D:\\CSVP2019\\copied\\annotations\\annotations_test_imc.csv'
trainFrames, trainLabels = parseAnnotationsCSV(trainAnnotationsCSVPath, trainFramesPath, targetSize)

trainLabels = to_categorical(trainLabels)


######### MODEL TRAINING #########

history = model.fit(
    trainFrames,
    trainLabels,
    callbacks=[checkpoint],
    epochs=10,
    validation_split=0.1
)
