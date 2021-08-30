import numpy as np
import os
import cv2 as cv
import random
import pickle
import pathlib



def create_dataset():
    base_directory = pathlib.Path(__file__).parent.absolute()
    os.chdir(base_directory)

    while True:
        try:
            trainOrTest = int(input('Do you want to create a training or testing dataset? (0 = testing, 1 = training): '))
            
            if trainOrTest in [0, 1]:
                break
        except ValueError as e:
            print(f'{e}, Please enter proper values!')

    folders = ['test', 'train']
    trainOrTest = folders[trainOrTest]


    training_data = create_training_data()
    random.shuffle(training_data)
    print('Length of the training data is', len(training_data))

    print('image size', len(training_data[0][0]))


    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    IMG_SIZE = len(training_data[0][0])
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    for i in range(len(X)):
        print(i+1, X[i].shape)
    print(y)
    pickle_out = open('datasets/X' + trainOrTest + '.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('datasets/y' + trainOrTest + '.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()

def create_training_data():
    datadir = os.path.join(baseDirectory, 'datasets', trainOrTest)
    categories = ['0', '1']
    training_data = []
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img))
                img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
                training_data.append([img_array, class_num])
            except Exception as e:
                pass
    return training_data
    