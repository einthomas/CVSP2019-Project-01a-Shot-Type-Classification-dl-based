from tensorflow import keras
from .util import *


def loadModel():
    if os.path.exists(getConfigRelativePath('modifiedModel')):
        model = keras.models.load_model(getConfigRelativePath('modifiedModel'))
    else:
        # Load pretrained model for transfer learning
        oldModel = keras.models.load_model(getConfigRelativePath('originalModel'))

        # Discard the last two layers (global avg pooling and the last dense layer)
        layers = oldModel.layers[len(oldModel.layers) - 3].output

        # Add two new layers
        layers = keras.layers.GlobalAveragePooling2D()(layers)
        layers = keras.layers.Dense(4, activation='softmax')(layers)

        # Replace the input layer to change the input shape
        oldModel._layers[0]._batch_input_shape = (None, 224, 224, 3)

        # Build new model
        model = keras.models.Model(inputs=oldModel.input, outputs=layers)

        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-4),
            # optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.save(getConfigRelativePath('modifiedModel'))

    with open(os.path.join(getConfigRelativePath('commonLogs'), 'modelSummary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model
