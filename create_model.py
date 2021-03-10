import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Only solution for an error which occured when training the model, deals with gpu memory allocation
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


X = pickle.load(open())
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

get_label_name = metadata.features['label'].int2str     # creates a function object that we can use to get labels

# display 2 images from the dataset
# for image, label in train.take(2):
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))

# plt.show()

# for img, label in raw_train.take(2):
#     print("Original shape:", img.shape)

# for img, label in train.take(2):
#     print("New shape:", img.shape)
IMG_SIZE = 160
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

#Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')



for image, _ in train_batches.take(1):
    pass

feature_batch = base_model(image)
print(feature_batch.shape)

base_model.trainable = False


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

print(model.summary())

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Evaluate the model now to see how it performs before training
initial_epochs = 3
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

# Now train it on new images
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)





# (train_images, train_labels), (test_images, test_labels) = 

# train_images, test_images = train_images / 255.0, test_images / 255.0

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1))

# print(model.summary())

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10,
#                     validation_data=(test_images, test_labels))

# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# print(test_acc)