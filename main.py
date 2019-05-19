from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
import pandas as pd

# Paths to data .csv
data_train_file = "..\\KerasFirstProject\\Datasets\\mnist_train.csv"
data_test_file = "..\\KerasFirstProject\\Datasets\\mnist_test.csv"

# Load files
df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)

# Prints first n rows, default 5.
print(df_train.head())

# Load data, the first column has labels
train_features = df_train.values[:, 1:] / 255
train_labels = df_train['label'].values

# Turns labels into array of len 10 with zeros except one value which indicates the label.
train_labels = tf.keras.utils.to_categorical(train_labels)

# Creating network
model = Sequential()
model.add(Dense(30, activation=tf.nn.relu, input_shape=(784,)))
model.add(Dense(20, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

EPOCHS = 2
BATCH_SIZE = 30

model.fit(train_features, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
