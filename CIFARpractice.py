import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test,y_test) =  cifar10.load_data()

x_train.shape

x_train[0].shape

# plt.imshow(x_train[89])
# shows the picture in the dataset

x_train[0].max()

x_train = x_train/255

x_test = x_test/255

# y_test

from tensorflow.keras.utils import to_categorical

y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# CONVOLUTIONAL LAYER

model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape = (32,32,3), activation='relu'))

# POOLING LAYER
model.add(MaxPool2D(pool_size=(2,2)))

# CONVOLUTIONAL LAYER 2

model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape = (32,32,3), activation='relu'))

# POOLING LAYER 2
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x_train,y_cat_train, epochs=15,validation_data=(x_test, y_cat_test), callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)
metrics.columns
metrics[['acc', 'val_acc']].plot()
metrics[['loss','val_loss']].plot()
model.evaluate(x_test, y_cat_test, verbose=0)
metrics

from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict_classes(x_test)

print(classification_report(y_test, predictions))

import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True)

my_image = x_test[16]
plt.imshow(my_image)
y_test[16]

model.predict_classes(my_image.reshape(1,32,32,3))
