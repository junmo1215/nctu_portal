# coding = UTF8

"""
训练神经网络

e.g.
    python train_model.py 0
"""

import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator


BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 5

def read_data():
    data = []
    labels = []
    for image_file in paths.list_images(INPUT_DATA_FOLDER):

        # Load the image and convert it to grayscale
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize the letter so it fits in a 20x20 pixel box
        # image = resize_to_fit(image, 20, 20)
        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)
        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]
        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)
    
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels

def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax', name='preds'))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
    return model

def main():
    data, labels = read_data()

    (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=0)

    # print(np.array(X_train).shape, np.array(X_test).shape)
    # print(np.array(Y_train).shape, np.array(Y_test).shape)

    lb = LabelBinarizer().fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)

    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    model = build_model()
    model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_test, y_test))
    model.save(MODEL_FILENAME)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("captcha_type", type=int,
                        default=-1,
                        help='captcha_type, value in {0, 1, 2, 3}')
    args = parser.parse_args()

    CAPTURE_TYPE = args.captcha_type

    INPUT_DATA_FOLDER = os.path.join("data", "single_letters", "{}".format(CAPTURE_TYPE))
    MODEL_FILENAME = os.path.join("model", "captcha_model{}.hdf5".format(CAPTURE_TYPE))
    MODEL_LABELS_FILENAME = os.path.join("model", "model_labels{}.dat".format(CAPTURE_TYPE))

    main()

