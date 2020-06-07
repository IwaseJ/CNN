##############################################
# Data Processing
##############################################

import os

data_dir = 'D:\\Python\\DeepLearning\\Tensorflow 2 and Keras Deep Learning Bootcamp\\cell_images'

# Shows the folders within the path
os.listdir(data_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread

test_path = data_dir + '\\test\\'
train_path = data_dir + '\\train\\'

# Shows the folders within the path
os.listdir(test_path)
os.listdir(train_path)

# Shows the image
os.listdir(train_path + 'parasitized')[0]

# View the images
para_cell = train_path + 'parasitized\\' + 'C100P61ThinF_IMG_20150918_144104_cell_162.png'

imread(para_cell)

imread(para_cell).shape

plt.imshow(imread(para_cell))

os.listdir(train_path + 'uninfected')[0]
uninfected_cell = train_path + 'uninfected\\' + 'C100P61ThinF_IMG_20150918_144104_cell_128.png'

uninfected_cell = imread(uninfected_cell)
plt.imshow(uninfected_cell)

# See how many images are in the folder

#len(os.listdir(train_path + 'parasitized'))
#len(os.listdir(train_path + 'uninfected'))
#len(os.listdir(test_path + 'parasitized'))
#len(os.listdir(test_path + 'uninfected'))

# Find the average shape of the images

dim1 = []
dim2 = []

# For loop reading all the images to check image shape
for image_filename in os.listdir(test_path + 'uninfected'):
  img = imread(test_path + 'uninfected\\' + image_filename)
  d1, d2, colors = img.shape
  # Saving the dimensions and appending them to dim1 and dim2
  dim1.append(d1)
  dim2.append(d2)

# Check and plot the sizes
dim1
dim2
sns.jointplot(dim1, dim2)
np.mean(dim1)
np.mean(dim2)

# Mean sizes for both Dim1 and Dim2 is 130 and color channel is 3 (RGB)
image_shape = (130,130,3)

# There is too much data points to read all at once, need to break them up into batches
# 130*130*3 = 50700 data points to read

# Artificially expanding data set by manipulating images by rotating and transforming them
# CNN needs thousands and thousands of images to perform really well

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ranges are in %
# 0.1 is equal to 10%
image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip = True,
                               fill_mode='nearest')

# How the picture would look without the Data Generator
# plt.imshow(uninfected_cell)

para_img=imread(para_cell)

plt.imshow(para_img)

# After Data Generator randomly transformed
plt.imshow(image_gen.random_transform(para_img))

# Flow batches from directory
# In order to use flow_from_directory, you must organize the images in sub-directories
# Need to have a folder (sub-directory) for each class you want to test
# Using Uninfected and Parasitized in this model

image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)

##############################################
# Creating the Model
##############################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

model = Sequential()

# The more complex of a task you are dealing with, the more Convolutional Layers you should have
model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape = image_shape, activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size=(3,3), input_shape = image_shape, activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size=(3,3), input_shape = image_shape, activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stop  = EarlyStopping(monitor = 'val_loss', patience = 2)

batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size = image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')
# When training, you want to shuffle, when testing you do not want to shuffle
test_image_gen = image_gen.flow_from_directory(test_path,
                                                target_size = image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary',
                                                shuffle = False)
train_image_gen.class_indices

results = model.fit_generator(train_image_gen, epochs = 20,
                              validation_data=test_image_gen,
                              callbacks = [early_stop])

# Run the training model but for the class we used the pre-ran data
# It would take too long on a normal hardware computer

from tensorflow.keras.models import load_model

model_path = 'D:\\Python\DeepLearning\\Tensorflow 2 and Keras Deep Learning Bootcamp\\CNN\\malaria_detector.h5'
model = load_model(model_path, 
                   custom_objects = None,
                   compile = False)
model.summary()

##############################################
# Evaluating the Model
##############################################

pred = model.predict_generator(test_image_gen)

predictions = pred > 0.5

predictions

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(test_image_gen.classes, predictions))

confusion_matrix(test_image_gen.classes, predictions)

from tensorflow.keras.preprocessing import image

my_image = image.load_img(para_cell, target_size = image_shape)

my_image

my_image_arr = image.img_to_array(my_image)

my_image_arr.shape

my_image_arr = np.expand_dims(my_image_arr,axis=0)

my_image_arr.shape

model.predict(my_image_arr)
