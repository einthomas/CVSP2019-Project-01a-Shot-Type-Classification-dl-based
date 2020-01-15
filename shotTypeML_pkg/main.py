import sys
import getopt
import cv2

from Common.imageUtil import loadImage, loadImagesFromFolder, preprocessImage

from Develop.predict import *
from PIL import Image

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

def printUsage():
    print('usage: ' + sys.argv[0] + ' [-h] [-i path] [-v path] [-o path]')
    print('-h: prints usage information')
    print('-i path: specifies a path where an input image or a folder containing images is located')
    print('-v path: specifies a path where an input video or a folder containing videos is located')
    #print('-o path: specifies an output path where the annotated image or video (.mp4) is stored')


def main(argv):
    # Parse command line arguments
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

    inputPaths = []
    isFolderInput = os.path.isdir(inputPath)
    if isFolderInput:
        inputPaths = [os.path.join(inputPath, file) for file in os.listdir(inputPath)]
    else:
        inputPaths.append(inputPath)

    targetImageSize = 112 # TODO CHANGE TO 224
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
    csvFile = open(outputPath, "w")
    csvFile.write('\n'.join(csvContent))
    csvFile.close()


    '''
    isVideoInput = False
    if inputPath.endswith('.mp4'):
        isVideoInput = True
    if isVideoInput and outputPath == '':
        print(str('classifying a video requires to specify an output path [-o path]'))
        sys.exit(2)

    isFolderInput = os.path.isdir(inputPath)

    targetImageSize = 112       # TODO CHANGE TO 224
    labels = []
    if isVideoInput:
        videoPaths = []
        for f in os.listdir(inputPath):
            if os.path.isfile(f) and f.endswith(".mp4"):
                videoPaths.append(f)
        if isFolderInput:
            images = loadImagesFromFolder(inputPath, targetImageSize, True)
        else:
            images.append(loadImage(inputPath, targetImageSize, True))
    else:
        images = []
        if isFolderInput:
            images = loadImagesFromFolder(inputPath, targetImageSize, True)
        else:
            images.append(loadImage(inputPath, targetImageSize, True))
        labels = predictShotType_production(images)
    '''

if __name__ == '__main__':
    main(sys.argv[1:])
