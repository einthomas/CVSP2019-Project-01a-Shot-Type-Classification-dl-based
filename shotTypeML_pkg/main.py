import sys
import getopt

from Common.imageUtil import loadImage, loadImagesFromFolder

from Develop.predict import *

# Fix "failed to initialize cuDNN" by explicitly allowing to dynamically grow
# the memory used on the GPU
# https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464957482
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

def printUsage():
    print('usage: ' + sys.argv[0] + ' [-h] [-c path] [-i path] [-o path]')
    print('either specify a config file via -c or input and output paths manually via -i and -o')
    print('-c path: configuration is read from the specified yaml located at path')
    print('-h: prints usage information')
    print('-i path: specifies a path where an input image or video (.mp4) or a folder containing images or videos (.mp4) is located')
    print('-o path: specifies an output path where the annotated image or video (.mp4) is stored')


def main(argv):
    # Parse command line arguments
    inputPath = ''
    outputPath = ''
    try:
        opts, args = getopt.getopt(argv, 'hi:o:', ['help'])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    for option, argument in opts:
        if option in ('-h', '--help'):
            printUsage()
            sys.exit()
        elif option == '-i':
            inputPath = argument
        elif option == '-o':
            outputPath = argument
        else:
            assert False, 'unhandled option'

    isVideoInput = False
    if inputPath.endswith('.mp4'):
        isVideoInput = True
    if isVideoInput and outputPath == '':
        print(str('classifying a video requires to specify an output path [-o path]'))
        sys.exit(2)

    isFolderInput = os.path.isdir(inputPath)

    targetImageSize = 112       # TODO CHANGE TO 224
    images = []
    if isFolderInput:
        images = loadImagesFromFolder(inputPath, targetImageSize, True)
    else:
        images.append(loadImage(inputPath, targetImageSize, True))

    labels = predictShotType_production(images)
    print(labels)


if __name__ == '__main__':
    main(sys.argv[1:])
