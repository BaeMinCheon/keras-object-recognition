import keras
import cv2
# import sklearn.preprocessing
import sklearn.model_selection
import numpy as np
import random
import imutils.paths
import os

import facemodel

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
    image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
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

# (trainX, testX, trainY, testY) = sklearn.model_selection.train_test_split(data, labels, test_size=0.2, random_state=20)
# augment = keras.preprocessing.image.ImageDataGenerator()
# model = facemodel.setup()
# optimizer = keras.optimizers.Adam(lr=learning_rate, decay=learning_rate / epochs)