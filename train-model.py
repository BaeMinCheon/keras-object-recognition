import matplotlib
matplotlib.use('Agg')

import keras
import cv2
import sklearn.preprocessing
import sklearn.model_selection
import matplotlib.pyplot as plt
import numpy as np
import random
import imutils.paths
import pickle
import os

import MyNetwork

epochs = 20
learning_rate = 1e-3
batch_size = 20
image_dimension = (192, 192, 3)

current_directory = os.getcwd()
dataset_directory = os.path.join(current_directory, 'dataset')
datum_paths = sorted(list(imutils.paths.list_images(dataset_directory)))

random.seed(42)
random.shuffle(datum_paths)

data = []
labels = []

for path in datum_paths:
    image = cv2.imread(path)
    image = cv2.resize(image, dsize=(192, 192))
    image = keras.preprocessing.image.img_to_array(image)
    data.append(image)

    name = path.split(os.path.sep)[-2].split()
    labels.append(name)

data = np.array(data, 'float') / 255.0
labels = np.array(labels)

MLB = sklearn.preprocessing.MultiLabelBinarizer()
labels = MLB.fit_transform(labels)
for (i, label) in enumerate(MLB.classes_):
    print("{} : {}".format(i + 1, label))

(trainX, testX, trainY, testY) = sklearn.model_selection.train_test_split(data, labels, test_size=0.2, random_state=20)
augment = keras.preprocessing.image.ImageDataGenerator()
model = MyNetwork.Build(_wid=image_dimension[1], _hei=image_dimension[0], _dep=image_dimension[2], _cls=len(MLB.classes_), _act='sigmoid')
optimizer = keras.optimizers.Adam(lr=learning_rate, decay=learning_rate / epochs)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

hypothesis = model.fit_generator(augment.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY), steps_per_epoch=int(len(trainX) / batch_size), epochs=epochs, verbose=1)
model.save(os.path.join(current_directory, 'MyNetwork.model'))
MLB_file = open(os.path.join(current_directory, 'MLB.pickle'), 'wb')
MLB_file.write(pickle.dumps(MLB))
MLB_file.close()

plt.style.use('ggplot')
plt.figure()

plt.plot(np.arange(0, epochs), hypothesis.history['loss'], label='train loss')
plt.plot(np.arange(0, epochs), hypothesis.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, epochs), hypothesis.history['acc'], label='train accuracy')
plt.plot(np.arange(0, epochs), hypothesis.history['val_acc'], label='validation accuracy')
plt.title('training loss & accuracy')
plt.xlabel('epoch number')
plt.ylabel('loss (or accuracy)')
plt.legend(loc='upper left')
plt.savefig(os.path.join(current_directory, 'plot.png'))