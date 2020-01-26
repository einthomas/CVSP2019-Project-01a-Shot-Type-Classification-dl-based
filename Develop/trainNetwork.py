import yaml
from argparse import ArgumentParser
import datetime
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

from Common.imageUtil import *
from Common.lr_finder import LRFinder
from Develop.loadModel import *

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
tfConfig = tf.compat.v1.ConfigProto()
tfConfig.gpu_options.allow_growth = True


def trainNetwork(trainDataPath, valDataPath, logsPath, modelPath, modelWeightsPath, targetImageSize, epochs, useLRFinder):
    # Load path of training and validation images
    trainFramesPath = trainDataPath
    valFramesPath = valDataPath

    shotTypes = ['CU', 'MS', 'LS', 'ELS']

    # Load training data
    print("loading training data...")
    trainFrames, trainLabels = loadImagesAndLabels(trainFramesPath, shotTypes, targetImageSize)
    trainLabels = to_categorical(trainLabels)

    # Use data augmentation
    datagenTrain = ImageDataGenerator(brightness_range=[0.8, 1.0], samplewise_center=True, samplewise_std_normalization=True,
                                      width_shift_range = 0.2, height_shift_range = 0.05, horizontal_flip=True,
                                      fill_mode='reflect')
    datagenTrain.fit(trainFrames)

    # Load validation data
    print("loading validation data...")
    valFrames, valLabels = loadImagesAndLabels(valFramesPath, shotTypes, targetImageSize)
    valLabels = to_categorical(valLabels)

    # Use data augmentation
    datagenVal = ImageDataGenerator(brightness_range=[0.8, 1.0], samplewise_center=True, samplewise_std_normalization=True,
                                    width_shift_range = 0.2, height_shift_range = 0.05, horizontal_flip=True,
                                    fill_mode='reflect')
    datagenVal.fit(valFrames)

    # Create a new log directory
    logDir = os.path.join(logsPath, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logDir)
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=1)
    os.makedirs(os.path.join(logDir, 'train\plugins\profile'))

    # For simpler reproducibility of training results, all .py source files are copied to the Logs folder of the current
    # training
    # Recursively go through all files
    '''
    for root, directories, files in os.walk(configHandler.config['workingDirectory']):
        if "Logs" not in root:
            for file in files:
                # Go through all .py source files
                if file.endswith(".py"):
                    # Recreate the folder structure in the Logs folder
                    directory = root[root.rfind('\\') + 1:]
                    targetPath = os.path.join(logDir, 'src', directory)
                    if not os.path.exists(targetPath):
                        os.makedirs(targetPath)

                    # Copy the file
                    copyfile(os.path.join(root, file), os.path.join(targetPath, file))
    '''

    # Use ModelCheckpoint to save the weights whenever the validation loss is minimal
    modelCheckpoint = keras.callbacks.ModelCheckpoint(modelWeightsPath, save_weights_only=True,
                                                      monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    # Train the model
    model = loadModel(modelPath, modelWeightsPath, targetImageSize)

    # Write the model summary into modelSummary.txt
    with open(os.path.join(logDir, 'modelSummary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # During development the learning rate finder class by Bradley Kenstler has been used to find an optimal learning rate
    callbacks = [tensorboardCallback, modelCheckpoint]
    if useLRFinder:
        lr_finder = LRFinder(min_lr=1e-7, max_lr=1e-3, steps_per_epoch=np.ceil(len(trainFrames) / 32.0), epochs=epochs)
        callbacks.append(lr_finder)

    model.fit_generator(
        datagenTrain.flow(trainFrames, trainLabels, batch_size=32),
        validation_data=datagenVal.flow(valFrames, valLabels, batch_size=32),
        callbacks=callbacks,
        epochs=epochs,
        shuffle=True,
        steps_per_epoch=len(trainFrames) / 32.0
    )

    if useLRFinder:
        lr_finder.plot_loss()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', type=str, help='Config .yaml file containing configuration settings')
    args = parser.parse_args()

    with open(args.config) as configFile:
        config = yaml.full_load(configFile)

    trainNetwork(
        config['trainFrames'],
        config['valFrames'],
        config['logs'],
        config['model'],
        config['modelWeights'],
        int(config['targetImageSize']),
        int(config['trainingEpochs']),
        bool(config['useLRFinder'])
    )
