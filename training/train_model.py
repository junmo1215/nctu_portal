import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator


INPUT_DATA_FOLDER = os.path.join("single_letters", "test")
MODEL_FILENAME = os.path.join("model", "captcha_model.hdf5")
MODEL_LABELS_FILENAME = os.path.join("model", "model_labels.dat")


# initialize the data and labels
data = []
labels = []

# def test(X):
#     for i in range(28):
#         for j in range(28):
#             print(X[i, j], end="")
#         print("")

# loop over the input images
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


# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# print("X_train.shape:{}\tX_test.shape:{}", np.array(X_train).shape, np.array(X_test).shape)
# print("Y_train.shape:{}\tY_test.shape:{}", np.array(Y_train).shape, np.array(Y_test).shape)


# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

model = Sequential()

model.add(Conv2D(32, (5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="relu"))
adam = optimizers.Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# set callback
tb_cb = TensorBoard(log_dir="./log", histogram_freq=0)
# change_lr = LearningRateScheduler(scheduler)
cbks = [tb_cb]

# using real-time data augmentation
datagen = ImageDataGenerator(horizontal_flip=False,
            width_shift_range=0.05,height_shift_range=0.05,
            fill_mode='constant',cval=0.)
datagen.fit(X_train)

# start traing 
model.fit_generator(datagen.flow(X_train, Y_train,batch_size=128),
                    steps_per_epoch=55,
                    epochs=50,
                    callbacks=cbks,
                    validation_data=(X_test, Y_test))

# # Train the neural network
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=128, epochs=50, verbose=1, callbacks=cbks)

# Save the trained model to disk
model.save(MODEL_FILENAME)
