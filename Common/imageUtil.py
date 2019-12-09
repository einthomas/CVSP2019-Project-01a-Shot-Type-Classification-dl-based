import os
import numpy as np
from keras.preprocessing import image


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
