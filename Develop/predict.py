import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.utils import to_categorical
import os
import csv
import numpy as np

from Common.loadModel import loadModel

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True


######### MODEL LOADING #########
model = loadModel()


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


valAnnotationsCSVPath = 'D:\\CSVP2019\\copied_merge\\annotations\\annotations_val.csv'
valFramesPath = 'D:\\CSVP2019\\copied_merge\\data\\val'
valFrames, valLabels = parseAnnotationsCSV(valAnnotationsCSVPath, valFramesPath, targetSize)
# valLabels = to_categorical(valLabels)

a = model.predict(valFrames)
total = len(a)
right = 0
for i in range(0, len(a)):
    prediction = np.argmax(a[i])
    gt = valLabels[i]
    if prediction == gt:
        right += 1
    print(prediction)
    print(gt)
print(total)
print(right)

# a.to_csv('D:\\CSVP2019\\copied_merge\\asdf.csv')
