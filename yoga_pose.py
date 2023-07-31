import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os

# set the random seed
seed = 314
tf.random.set_seed(seed)
np.random.seed(seed)

train_path = "DATASET/TRAIN/"
test_path = "DATASET/TEST/"

class_names = []
train_distrib = []
test_distrib = []

# for train
for dir in os.listdir(train_path):
    class_names.append(dir)  # append the folder name i.e the class
    number = len(os.listdir(train_path + dir + '/'))  # count the files on the dir
    train_distrib.append(number)  # append

# for test
for dir in os.listdir(test_path):
    class_names.append(dir) # append the folder name i.e the class
    number = len(os.listdir(test_path + dir + '/'))  # count the files on the dir
    test_distrib.append(number)  # append

# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.pie(train_distrib, labels=class_names)
# plt.title(f'Train - {sum(train_distrib)} files')
# plt.subplot(1, 2, 2)
# plt.pie(test_distrib, labels=class_names)
# plt.title(f'Test - {sum(test_distrib)} files')
# plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1/255, # Normalize the pixel values to [0, 1]
    shear_range = 0.2, # it means to cut the image
    zoom_range = 0.2, # make zoom
    horizontal_flip = True # invert horizontally
)

test_datagen = ImageDataGenerator(
    rescale = 1/ 255,
    horizontal_flip = True,
    validation_split=0.25, # Split the data for validation (25%)
)

###########################
image_size = 128

# and load the images
train_gen = train_datagen.flow_from_directory(
    train_path, # the data folder
    batch_size = 16,
    class_mode = 'categorical', # for multi class
    target_size = (image_size, image_size), # the image size
    shuffle = True,
    seed = 314
)


test_gen = test_datagen.flow_from_directory(
    test_path, # the data folder
    batch_size = 16,
    class_mode = 'categorical', # for multi class
    target_size = (image_size, image_size), # the image size
    shuffle = True,
    seed = seed,
    subset='training', # since test will be bigger than val
)

# now validation
val_gen = test_datagen.flow_from_directory(
    test_path, # the test folder
    batch_size = 16,
    class_mode = 'categorical', # for multi class
    target_size = (image_size, image_size), # the image size
    shuffle = True,
    seed = seed,
    subset='validation', # since test will be bigger than val
)

from tensorflow.keras import Sequential, layers

num_classes = len(class_names)

model = Sequential([
    # the imput shape must be defined
    layers.InputLayer(input_shape=[image_size, image_size, 3]),
    layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),  # this layer turn off random neurons each step
    # it helps to improve the model and helps to prevent overfitting
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
 layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
##################################################################
from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(
    min_delta=0.005, #minimum change in the monitored metric to be considered as an improvement.
    patience=5, # number of epochs with no improvement after which training will be stopped.
    restore_best_weights=True #weights of model at epoch with best performance will be restored before training stop
)
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

history = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs = 25,
    callbacks = None
)

plt.figure(figsize=(15, 10))

# plot the loss function
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss Function')
plt.grid(True)
plt.legend()

# and the accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.grid(True)
plt.title('Accuracy')
plt.legend()

plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# define the matrix with the real classes and the predicted
m = confusion_matrix(test_gen.classes, y_pred)
# the labels for the plot
labels = class_names
plt.figure(figsize=(20, 8))
# create the plot
heatmap = sns.heatmap(m, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', color='blue')
# labels for the axes
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

