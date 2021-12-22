import numpy as np
import os
import tensorflow as tf

from PIL import Image
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from keras import utils

def load_dataset(dir):
    x, y = list(), list()

    for subdir in os.listdir(dir):
        image_path = dir + subdir + '/'

        if not os.path.isdir(image_path):
            continue

        images = load_images(image_path)
        labels = [subdir for _ in range(len(images))]

        x.extend(images)
        y.extend(labels)
    
    return np.asarray(x), np.asarray(y)

def load_images(dir):
    images = list()

    for filename in os.listdir(dir):
        image_path = dir + filename

        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((160, 160))
        arr_image = np.asarray(image)

        images.append(arr_image)

    return images

train_x, train_y = load_dataset('data50/train/')
train_x_flatten = train_x.reshape(train_x.shape[0], -1)

encoder = preprocessing.LabelEncoder()
encoder.fit(train_y)
encoded_y = encoder.transform(train_y)

y = utils.np_utils.to_categorical(encoded_y)

nodeLayer1 = 20
nodeLayer2 = 10
nodeLayer3 = 10

model = keras.Sequential(
    [
        layers.Dense(nodeLayer1, input_dim=len(train_x_flatten[0]), activation='relu'),
        layers.Dense(nodeLayer2, activation = 'relu'),
        layers.Dense(nodeLayer3, activation='relu'),
        layers.Dense(len(y[0]), activation='softmax')
    ]
)

opt = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.fit(train_x_flatten, y, epochs = 1000, batch_size = 150)

test_x, test_y = load_dataset('data/test/')

test_x_flatten = test_x.reshape(test_x.shape[0], -1)

encoder = preprocessing.LabelEncoder()
encoder.fit(test_y)
encoded_y_test = encoder.transform(test_y)

y_test = utils.np_utils.to_categorical(encoded_y_test)

y_predict = model(test_x_flatten)


print(np.round(y_predict))

_, accuracy = model.evaluate(test_x_flatten, y_test)
