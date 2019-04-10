# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:08:14 2019

@author: Administrator
"""

from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# import the Fasion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Each image is mapped to a single label. 
#Since the class names are not included with the dataset, 
#store them here to use later when plotting the images:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandel', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Let's explore the format of the dataset before training the model. 
#The following shows there are 60,000 images in the training set, 
#with each image represented as 28 x 28 pixels:
print(train_images.shape)
print(train_labels.shape)

# Preprocess the data
plt.figure()
plt.imshow(train_images[10]);
plt.colorbar()
plt.grid(False);
plt.show()

#scale the images
train_images = train_images/255.0
test_images = test_images/255.0

# display the first 25 images from the training set
# and display the class name below each image
"""
plt.figure(figsize=(28,28))
for i in range(25):
    plt.subplot(5,5,i+1)
    #plt.xticks([])
    #plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()   # may be this line can be del
"""
#Build the model
#Set up the layers
model = keras.Sequential([
        keras.layers.Flatten(input_shape = (28, 28)),
        keras.layers.Dense(128, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
        ])
"""
The first layer in this network, tf.keras.layers.Flatten, transforms the format 
of the images from a two-dimensional array (of 28 by 28 pixels) to a 
one-dimensional array (of 28 * 28 = 784 pixels). Think of this layer as 
unstacking rows of pixels in the image and lining them up. This layer has no 
parameters to learn; it only reformats the data.

After the pixels are flattened, the network consists of a sequence of two 
tf.keras.layers.Dense layers. These are densely connected, or fully connected, 
neural layers. The first Dense layer has 128 nodes (or neurons). 
The second (and last) layer is a 10-node softmax layer that returns an array of
 10 probability scores that sum to 1. Each node contains a score that indicates
 the probability that the current image belongs to one of the 10 classes.
"""
# Compile the model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#train the model
model.fit(train_images, train_labels, epochs = 5)

#Evaluate acc
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:',test_acc)

#Make predictions
predictions = model.predict(test_images)  # a (10000, 10) mat

# graph to look at the full set of 10 channels
def plot_image(i, predictions_array, true_label, img):
    predictions_array = predictions_array[i]
    true_label = true_label[i]
    img = img[i] 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
      
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
      
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
     
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
#Let's look at the 0th image, predictions, and prediction array
i = 0
plt.figure(figsize = (6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize = (6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

# plot the first X test images, predicted labels, and the true labels
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

"""
tf.keras models are optimized to make predictions on a batch, or collection, 
of examples at once. Accordingly, even though we're using a single image, we 
need to add it to a list:
"""
img = test_images[0]
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = model.predict(img)
print(predictions_single)


#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.








    






















