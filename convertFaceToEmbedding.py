import pickle
import numpy as np
import tensorflow as tf

# get face embedding for one face
def get_embedding(model, images):
    images = images.astype('float32')
    mean, std = images.mean(), images.std()
    images = (images - mean) / std
    samples = np.expand_dims(images, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# load face datasets
X = np.array(pickle.load(open('data/datasets/Xtrain.pickle', 'rb')))
y = np.array(pickle.load(open('data/datasets/ytrain.pickle', 'rb')))


x_test = np.array(pickle.load(open('data/datasets/Xtest.pickle', 'rb')))
y_test = np.array(pickle.load(open('data/datasets/ytest.pickle', 'rb'))) 

# load facenet model
model = tf.keras.models.load_model('facenet_keras.h5')

# convert each face image into a face embedding
newX = []
for image in X:
    embedding = get_embedding(model, image)
    newX.append(embedding)
newX = np.asarray(newX)
print(newX.shape)

# do the same for the test set

newX_test = []
for image in x_test:
    embedding = get_embedding(model, image)
    newX_test.append(embedding)
newX_test = np.asarray(newX_test)
print(newX_test.shape)

np.savez_compressed('embeddedFaceData.npz', newX, y, newX_test, y_test)
