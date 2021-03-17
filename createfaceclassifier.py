import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import pickle

data = np.load('embeddedFaceData.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']


in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

model = SVC(kernel='linear')
model.fit(trainX, trainy))
pickle_out = open('myFaceClassifier.pickle', 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()

yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
print(f'Accuracy: train={score_train*100:.3f}, test={score_test*100:.3f}')