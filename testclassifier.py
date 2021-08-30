from random import choice
import numpy as np
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import pickle

# load faces
data = np.array(pickle.load(open('data/datasets/Xtest.pickle', 'rb')))
testX_faces = data
# load face embeddings
data = load('embeddedFaceData.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# test model on 5 random examples from the test dataset
for i in range(5):
    selection = choice([j for j in range(testX.shape[0])])
    randomImage = testX_faces[selection]
    face_embed = testX[selection]
    face_class = testy[selection]
    face_name = out_encoder.inverse_transform([face_class])
    # prediction for the face
    samples = expand_dims(face_embed, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print(f'Predicted: {predict_names[0]} ({class_probability:.3f})')
    print(f'Expected: {face_name[0]}')
    pyplot.imshow(randomImage)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()