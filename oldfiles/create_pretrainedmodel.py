import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
import pickle
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


IMG_SIZE = 160  # All images will be resized to 160x160

def format_example(image, label):
    """
    returns an image that is reshaped to IMG_SIZE
    """

    image = tf.cast(image, tf.float32)
    image = (image/255)
    return image, label

X = np.array(pickle.load(open('data/datasets/Xtrain.pickle', 'rb')))
y = np.array(pickle.load(open('data/datasets/ytrain.pickle', 'rb')))


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

#Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.models.load_model('facenet_keras.h5')
base_model.trainable = False
print(base_model.summary())

first = tf.keras.Input(shape=(128),)

model = keras.Sequential(
    base_model,
    [
        keras.Input(shape=(128,)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=10),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)
print(model.summary())
# base_learning_rate = 0.0001
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# # train model on new images
# history = model.fit(X, y,
#                     epochs=3,
#                     validation_split=0.1)

# acc = history.history['accuracy']
# print(acc)


