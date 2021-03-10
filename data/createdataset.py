import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import random
import pickle
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


DATADIR = 'C:/Users/14694/MPS/tf-venv/Facial Recognition CNN/FaceDataProcessing/Datasets/'
CATEGORIES = ['0', '1']

# img_array = cv.imread(os.path.join(DATADIR + CATEGORIES[0],'IMG_01.jpg'), cv.IMREAD_GRAYSCALE)
# plt.imshow(img_array, cmap='gray') 
# print(img_array)
# plt.show()

# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)
#     for img in os.listdir(path):
#         img_array = cv.imread(os.path.join(path,img), cv.IMREAD_GRAYSCALE)
#         plt.imshow(img_array, cmap='gray') 
#         plt.show()
#         break
#     break

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img), cv.IMREAD_GRAYSCALE)
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

create_training_data()
# random.shuffle(training_data)
print('Length of the training data is', len(training_data))

print('image size', len(training_data[0][0]))


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

IMG_SIZE = len(training_data[0][0])
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(X)
print(y)

# pickle_out = open('X.pickle', 'wb')
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open('y.pickle', 'wb')
# pickle.dump(y, pickle_out)
# pickle_out.close()