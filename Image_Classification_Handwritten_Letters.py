#Imports and utilities!
import tensorflow as tf
from keras.utils import to_categorical

#tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)

#Mnist is a dataset of handwritten letters, based on which i created, trained and tested an ANN!
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# returns train and test data of mnist hand written letters ;)

# Lets check how they look!
print('x_train_shape:', x_train.shape)
print('y_train_shape:', y_train.shape)
print('x_test_shape:', x_test.shape)
print('y_test_shape:', y_test.shape)

# 1st index = number of example images
# 2nd, 3rd index = pixels of each example image

# Let's plot an image for example:
# The first one is a handwritten number 5..

from matplotlib import pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(x_train[0], cmap='binary')
plt.show()

# Display Labels
y_train[0]
print(set(y_train))


# One Hot Encoding
# Transform output class "number 5 for example" to a list with 10 elements where the corresponding
# element is denoted with 1 and the rest with 0.

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Check new y_train,y_test shapes!
print('y_train_encoded shape: ', y_train_encoded.shape)
print('y_test_encoded shape: ', y_test_encoded.shape)

y_train_encoded[0]

# ANN!!!
# Convert inputs x_train, x_test to 2D vectors: x_train_reshaped, x_test_reshaped!

import numpy as np

x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

# Check if it worked!
print('x_train_reshaped shape: ', x_train_reshaped.shape)
print('x_test_reshaped shape: ', x_test_reshaped.shape)

# Now each element in the vector 784x1 corresponds to a pixel value in the image..

# Check random pixel values for a specific example:

print(set(x_train_reshaped[0]))

# Data Normalization: calculate mean and standard deviation for dataset and then normalize for faster computation time.
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)
print(x_mean, x_std)

epsilon = 1e-10

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

# Do not print, its too big!
# print(x_train_norm,x_test_norm)
# print(set(x_train_norm[0]))

# Create a ANN model with Keras and Tensorflow.
# I will use: 2 hidden layers with 128 nodes each and 1 output layer with 10 nodes (for the 10 classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Softmax output function due to the multiclass nature of the problem..
model = Sequential(
    [Dense(128, activation='relu', input_shape=(784,)), Dense(128, activation='relu'), Dense(10, activation='softmax')])

# Optimizer used: stochastic gradient descent (can also use adam!)
# Loss to be minimized: Categorical cross entropy..

model.compile(optimizer='sgd', loss='categorical_crossentropy', metric=['accuracy'])

model.summary()

# Training the Network Model !
# Use train set to train the model
model.fit(x_train_norm, y_train_encoded, epochs=3)

# Check  accuracy using the test set x and y !
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test Accuracy is: ', accuracy * 100)

# Predictions on Test Set
preds = model.predict(x_test_norm)
print('Shapre of predictions: ', preds.shape)


# Plots
plt.figure(figsize=(12, 12))

start_index = 0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred = np.argmax(preds[start_index + i])
    gt = y_test[start_index + i]

    col = 'g'
    if pred != gt:
        col = 'r'

    plt.xlabel('i={}, pred={}, gt={}'.format(start_index + i, pred, gt))
    plt.imshow(x_test[start_index + 1], cmap='binary')

plt.plot(preds[8])
plt.show()

