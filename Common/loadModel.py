from tensorflow import keras
from .util import *


def loadModel():
    baseModel = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze all layers except for the last eight
    for layer in baseModel.layers[:-8]:
        layer.trainable = False
    for layer in baseModel.layers[-8:]:
        layer.trainable = True

    for layer in baseModel.layers:
        print(layer, layer.trainable)

    print("building model...")
    x = baseModel.output
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    predictions = keras.layers.Dense(4, activation='softmax')(x)
    model = keras.models.Model(inputs=baseModel.input, outputs=predictions)

    print("compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(lr=5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    if os.path.exists(getConfigRelativePath('checkpointModel')):
        # Load weights
        print("load weights from " + getConfigRelativePath('checkpointModel'))
        model.load_weights(getConfigRelativePath('checkpointModel'))

    model.save(getConfigRelativePath('modifiedModel'))

    with open(os.path.join(getConfigRelativePath('commonLogs'), 'modelSummary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model
