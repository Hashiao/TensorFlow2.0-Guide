# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:59:46 2019

@author: Administrator

import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ys = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]
model.fit(xs, ys, epochs = 2000)
print(model.predict([7.0]))

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
mnist = tf.keras.datasets.mnist
" load and prepare the MNIST "
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

" Build the tf.keras.Sequential model by stacking layers. "
" Choose an optimizer and loss function used for training:"
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

"train and evaluate model"
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)



    