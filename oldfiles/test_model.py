import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

x_test = np.array(pickle.load(open('data/datasets/Xtest.pickle', 'rb')))
y_test = np.array(pickle.load(open('data/datasets/ytest.pickle', 'rb')))
model = tf.keras.models.load_model('myface')
print('length of the test dataset:', len(x_test))
predictions = model.predict(x_test)


count = 0
for x in range(len(predictions)):
    guess = (np.argmax(predictions[x]))
    actual = y_test[x]
    predictionText = ['not you', 'you']
    print('I predict this is', predictionText[guess])
    print('This is actually:', predictionText[actual])
    if guess != actual:
        print('---------------')
        print('WRONG')
        print('---------------')
        count+=1
    plt.imshow(x_test[x])
    plt.show()

print('The program got', count, 'wrong, out of', len(x_test))
print(str(100 - ((count/len(x_test))*100)) + '% correct')