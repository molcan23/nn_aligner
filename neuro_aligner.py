import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D

url1 = 'https://raw.githubusercontent.com/molcan23/nn_aligner/master/shuffle.csv'
url2 = 'https://raw.githubusercontent.com/molcan23/nn_aligner/master/target.csv'
url3 = 'https://raw.githubusercontent.com/molcan23/nn_aligner/master/test_X.csv'
url4 = 'https://raw.githubusercontent.com/molcan23/nn_aligner/master/test_Y.csv'

test_X = pd.read_csv(url3)
test_Y = pd.read_csv(url4)
shuffle = pd.read_csv(url1)
target = pd.read_csv(url2)

X = shuffle
y = target

classifier = Sequential()
classifier.add(Dense(X.shape[1], activation='relu', input_dim=X.shape[1]))
classifier.add(Dense(X.shape[1]+50, activation='tanh', input_dim=X.shape[1]))
classifier.add(Dense(X.shape[1], activation='relu', input_dim=X.shape[1]))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = classifier.fit(X, y, epochs=100, batch_size=10, validation_split=0.1)

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.figure()
plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.legend(loc='best')
plt.show()


score = classifier.evaluate(test_X, test_Y)
print("\n\nloss: {} | accuracy: {}".format(score[0], score[1]*100))

score = classifier.evaluate(test_X, np.ones(len(test_X)))
print("\n\nloss: {} | accuracy: {}".format(score[0], score[1]*100))