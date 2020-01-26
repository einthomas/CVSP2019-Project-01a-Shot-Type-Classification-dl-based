import cv2
import yaml
import os
import numpy as np
from shotTypeML_pkg.imageUtil import loadImage, preprocessImage
from shotTypeML_pkg.predict import predictShotType
from PIL import Image


class ShotTypeClassifierML:
    """ Interface for predicting the shot type of images and videos. """

    def __init__(self, configPath):
        """ Reads the provided .yaml config file. """

        with open(configPath) as configFile:
            self.config = yaml.full_load(configFile)



    def predict(self, inputPath, isVideoInput, csvOutputPath = ''):
        """ Predicts the shot type of the provided image(s) or video(s). If video(s) are provided, `isVideoInput`
         has to be set to `True`. The results are returned and stored if a path is provided (`csvOutputPath`) as CSV. """

        # Load input files (either a single image or video or all images or videos)
        # inside the specified folder
        inputPaths = []
        isFolderInput = os.path.isdir(inputPath)
        if isFolderInput:
            inputPaths = [os.path.join(inputPath, file) for file in os.listdir(inputPath)]
        else:
            inputPaths.append(inputPath)

        targetImageSize = int(self.config['targetImageSize'])
        csvContent = []
        if isVideoInput:  # Handle video input
            csvContent = ["videoPath;framePos;label"]
            # Go through all the videos, extract each frame, pre-process and
            # label them
            for i in inputPaths:
                # Open video
                cap = cv2.VideoCapture(i)
                while cap.isOpened():
                    # Extract frame
                    ret, image = cap.read()
                    if not ret:
                        break
                    framePos = cap.get(cv2.CAP_PROP_POS_FRAMES)

                    # Convert frame from a cv2 to a PIL image for preprocessing
                    pilImage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    pilImage = preprocessImage(pilImage, targetImageSize, True)

                    # Predict shot type
                    label = predictShotType(self.config['model'], self.config['modelWeights'], np.array([pilImage]))[0]

                    # Build CSV entry
                    csvContent.append(i + ";" + str(int(framePos)) + ";" + label)
                cap.release()
        else:  # Handle image input
            csvContent = ["imagePath;label"]
            # Go through all the images, pre-process and label them
            for i in inputPaths:
                # Predict shot type
                label = predictShotType(
                    self.config['model'],
                    self.config['modelWeights'],
                    loadImage(i, targetImageSize, True)
                )[0]

                # Build CSV entry
                csvContent.append(i + ";" + label)

        # Write CSV file if an output path is specified
        if csvOutputPath != '':
            csvFile = open(csvOutputPath, "w")
            csvFile.write('\n'.join(csvContent))
            csvFile.close()

        return csvContent
