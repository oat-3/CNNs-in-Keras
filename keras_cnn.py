# P10: CNNs in Keras
# keras_cnn.py
# Name: Oat (Smith) Sukcharoenyingyong
# Net ID: sukcharoenyi@wisc.edu
# CS login: sukcharoenyingyong


import tensorflow as tf
from tensorflow import keras
import pydot

from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# todo: takes an optional boolean argument and returns the data as described below
def get_dataset(training=True):
    # load the data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    #expand the train_images and test_images from 3D to 4D
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)
    # if training is true, return tuple of train images and labels
    # else return tuple of test images and labels
    if training:
        return (train_images, train_labels)
    else:
        return (test_images, test_labels)


# todo: takes no arguments and returns an untrained neural network as specified below
def build_model():
    model = keras.Sequential()
    # add two 2D convolutional layers
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    # flatten out the results into a single dimension
    model.add(Flatten())
    # add dense layer
    model.add(Dense(10, activation='softmax'))
    # compile
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# todo: takes the model produced by the previous function and the images and
#  labels produced by the first function and trains the data for T epochs;
#  does not return anything
def train_model(model, train_img, train_lab, test_img, test_lab, T):
    # transform the label data to a 2d array
    train_lab = keras.utils.to_categorical(train_lab)
    test_lab = keras.utils.to_categorical(test_lab)
    # train with validation data
    model.fit(x=train_img, y=train_lab, epochs=T, validation_data=(test_img, test_lab))


# todo: takes the trained model and test images,
#  and prints the top 3 most likely labels for the image at the given index,
#  along with their probabilities
def predict_label(model, images, index):
    # all of the probabilities of all of the images
    predictions = model.predict(images)
    # making a copy because I wanted to go through the data without changing anything
    # in the original predictions
    pred = predictions.copy()
    # list of the label from the class_names that we want
    im_list = []
    # list of percentage chance according to the label from class_names
    per_list = []

    # go through 3 times to select the three highest probabilities/percentages
    for i in range(3):
        max = float("-inf")
        maxIdx = -1
        # find the largest probability
        for j in range(len(pred[index])):
            if pred[index][j] > max:
                max = pred[index][j]
                maxIdx = j
        # add label and highest probability to the list
        im_list.append(class_names[maxIdx])
        per_list.append(pred[index][maxIdx])
        # change the highest probability to negative infinity, so that when we loop
        # through again we will get the next highest probability
        pred[index][maxIdx] = float("-inf")
    # turning decimals into percentages
    for i in range(len(per_list)):
        per_list[i] *= 100
    # print out the top 3 highest percentages and their respective class labels
    for i in range(len(im_list)):
        print(im_list[i] + ": " + str("{:.2f}".format(per_list[i])) + "%")


#(train_images, train_labels) = get_dataset()
# print(train_images.shape)
#
# (test_images, test_labels) = get_dataset(False)
# print(test_images.shape)
#
# model = build_model()
# keras.utils.plot_model(model, to_file='model.png')
#
# train_model(model, train_images, train_labels, test_images, test_labels, 5)
# predict_label(model, test_images, 0)