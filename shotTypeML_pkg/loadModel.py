import os
from tensorflow import keras


def loadModel(modelPath, modelWeightsPath, targetImageSize):
    """ Builds and returns a new VGG19 model with imagenet weights. The topmost
    layers are frozen and a dense layer with 4 outputs is appended. """

    # Use VGG19 with imagenet weights
    baseModel = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(targetImageSize, targetImageSize, 3))

    # Freeze all layers except for the last eight
    for layer in baseModel.layers[:-10]:
        layer.trainable = False
    for layer in baseModel.layers[-10:]:
        layer.trainable = True

    for layer in baseModel.layers:
        print(layer, layer.trainable)

    # Add a dropout, global avg pooling and dense layer
    print("building model...")
    x = baseModel.output
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    predictions = keras.layers.Dense(4, activation='softmax')(x)
    model = keras.models.Model(inputs=baseModel.input, outputs=predictions)

    # Compile model
    print("compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(lr=5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    # Load weights if they exist
    if os.path.exists(modelWeightsPath):
        print("load weights from " + modelWeightsPath)
        model.load_weights(modelWeightsPath)

    # Save the new model
    if not os.path.exists(os.path.dirname(modelWeightsPath)):
        os.makedirs(os.path.dirname(modelWeightsPath))
    model.save(modelPath)

    return model
