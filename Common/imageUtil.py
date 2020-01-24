import os

import numpy as np
from PIL import ImageOps
from keras.preprocessing import image


# Resize an image and apply a center crop. The returned image is a square with
# a width and height of targetSize
def centerCropImage(img, targetSize):
    # Resize image while keeping its aspect ratio
    width, height = img.size
    if height < targetSize:
        print(str(height) + " height < targetSize")
    aspectRatio = width / height
    resizedWidth = int(targetSize * aspectRatio)
    resizedWidth = resizedWidth + resizedWidth % 2
    img = img.resize((resizedWidth, targetSize))

    # Apply a center crop by cutting away the same number of pixels at both
    # sides, left and right, of the image
    width, height = img.size
    offsetX = round((width - targetSize) * 0.5)
    return img.crop((offsetX, 0, width - offsetX, height))


def preprocessImage(img, targetSize, standardize):
    img = img.convert('L')
    img = centerCropImage(img, targetSize)
    # img = ImageOps.autocontrast(img, cutoff=5)
    img = ImageOps.equalize(img, mask=None)
    img = image.img_to_array(img)
    img = img / 255.0

    # Reshape image from (224, 224, 1) to (224, 224, 3)
    img = np.squeeze(np.stack((img,) * 3, axis=-1))

    # Zero center normalization
    if standardize:
        img = (img - img.mean()) / img.std()

    return img


# Loads the images located at path. It is assumed that the images are located
# in folders named according to their shot type (CU, MS, LS or ELS)
def loadImagesAndLabels(path, shotTypes, targetSize, standardize=False):
    images = []
    labels = []

    for shotType in shotTypes:
        currentPath = os.path.join(path, shotType)
        for imageName in os.listdir(currentPath):
            labels.append(shotTypes.index(shotType))

            # Load and preprocess image
            img = image.load_img(os.path.join(currentPath, imageName), color_mode="grayscale")
            img = preprocessImage(img, targetSize, standardize)
            images.append(img)

    return np.array(images), np.array(labels)


# Loads all images located at path and in subfolders of path
def loadImagesFromFolder(path, targetSize, standardize=False):
    images = []

    for root, subdirs, files in os.walk(path):
        for file in files:
            # Load and preprocess image
            img = image.load_img(os.path.join(root, file), color_mode="grayscale")
            img = preprocessImage(img, targetSize, standardize)
            images.append(img)

    return np.array(images)


# Loads a single image located at path
def loadImage(path, targetSize, standardize=False):
    img = image.load_img(path)#, color_mode="grayscale")
    return np.array([preprocessImage(img, targetSize, standardize)])
