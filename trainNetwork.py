# tensorflow and keras
import tensorflow as tf
from tensorflow import keras

# Load pretrained model for transfer learning
oldModel = keras.models.load_model('D:\\CSVP2019\\model\\model_shotscale_967.h5')

# Discard the last two layers (global avg pooling and the last dense layer)
layers = oldModel.layers[len(oldModel.layers) - 3].output

# Add two new layers
layers = keras.layers.GlobalAveragePooling2D()(layers)
layers = keras.layers.Dense(4, activation='softmax')(layers)

# Build new model
model = keras.models.Model(inputs=oldModel.input, outputs=layers)
model.compile(
    #optimizer=keras.optimizers.Adam(lr=0.00001),
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='D:\\CSVP2019\\model\\model_transfer.h5',
    save_best_only=True
)
