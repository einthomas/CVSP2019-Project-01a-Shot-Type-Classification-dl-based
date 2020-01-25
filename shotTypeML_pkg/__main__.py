import os
import sys
import getopt
import cv2

from Common.imageUtil import loadImage, preprocessImage
from Common.util import config

from Develop.predict import predictShotType_production, predictShotType_testData
from PIL import Image

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True


def printUsage():
    print('usage: ' + sys.argv[0] + ' [-h] [-c path] [-i path] [-v path] [-o path]')
    print('-h: prints usage information')
    print('-c path: specifies a path to a .yaml config file')
    print('-i path: specifies a path to an input image or a folder containing images is located')
    print('-v path: specifies a path to an input video or a folder containing videos is located')
    print('-o path: specifies a CSV file where the results are stored')


def main(argv):
    # Parse command line arguments
    configPath = ''
    inputPath = ''
    outputPath = ''
    isVideoInput = False
    try:
        opts, args = getopt.getopt(argv, 'hi:v:o:', ['help'])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    for option, argument in opts:
        if option in ('-h', '--help'):
            printUsage()
            sys.exit()
        elif option == '-c':
            # Config input
            configPath = argument
        elif option == '-i':
            # Image input
            inputPath = argument
        elif option == '-v':
            # Video input
            inputPath = argument
            isVideoInput = True
        elif option == '-o':
            outputPath = argument
        else:
            assert False, 'unhandled option'

    if configPath == '':
        print('no config file specified, use argument -h to print usage information')
        sys.exit(2)

    if inputPath == '':
        print('no input files specified, use argument -h to print usage information')
        sys.exit(2)

    inputPaths = []
    isFolderInput = os.path.isdir(inputPath)
    if isFolderInput:
        inputPaths = [os.path.join(inputPath, file) for file in os.listdir(inputPath)]
    else:
        inputPaths.append(inputPath)

    targetImageSize = int(config['targetImageSize'])
    csvContent = []
    if isVideoInput:    # Handle video input
        csvContent = ["videoPath;framePos;label"]
        for i in inputPaths:
            # Open video
            cap = cv2.VideoCapture(i)
            while (cap.isOpened()):
                # Extract frame
                ret, image = cap.read()
                if not ret:
                    break
                framePos = cap.get(cv2.CAP_PROP_POS_FRAMES)

                # Convert frame from a cv2 to a PIL image for preprocessing
                pilImage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pilImage = preprocessImage(pilImage, targetImageSize, True)

                # Predict shot type
                label = predictShotType_production(np.array([pilImage]))[0]

                # Build CSV entry
                csvContent.append(i + ";" + str(int(framePos)) + ";" + label)
            cap.release()
    else:   # Handle image input
        csvContent = ["imagePath;label"]
        images = []
        for i in inputPaths:
            # Predict shot type
            label = predictShotType_production(loadImage(i, targetImageSize, True))[0]

            # Build CSV entry
            csvContent.append(i + ";" + label)

    # Write CSV file
    if outputPath != '':
        csvFile = open(outputPath, "w")
        csvFile.write('\n'.join(csvContent))
        csvFile.close()

    return csvContent


if __name__ == '__main__':
    main(sys.argv[1:])
