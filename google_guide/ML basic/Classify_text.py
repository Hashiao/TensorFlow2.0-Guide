# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:27:50 2019
@author: Administrator

This notebook classifies movie reviews as positive or negative using the text 
of the review. This is an example of binary—or two-class—classification, an 
important and widely applicable kind of machine learning problem.

We'll use the IMDB dataset that contains the text of 50,000 movie reviews from 
the Internet Movie Database. These are split into 25,000 reviews for training 
and 25,000 reviews for testing. The training and testing sets are balanced,
 meaning they contain an equal number of positive and negative reviews.

This notebook uses tf.keras, a high-level API to build and train models in 
TensorFlow. For a more advanced text classification tutorial using tf.keras, 
see the MLCC Text Classification Guide.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)

# Download the IMDB dataset
"""
The IMDB dataset comes packaged with TensorFlow. It has already been 
preprocessed such that the reviews (sequences of words) have been converted 
to sequences of integers, where each integer represents a specific word in 
a dictionary.

The argument num_words=10000 keeps the top 10,000 most frequently occurring
 words in the training data. The rare words are discarded to keep the size of 
 the data manageable.
"""
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

#Explore the data
print("Training entries:{}, labels:{}".format(len(train_data), len(train_labels)))
# train_data[0] is a review of movie
print(train_data[5])
print(len(train_data[5]))

# Convert the integers back to words
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# the first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(train_data[0])

#Prepare the data
"""
The reviews—the arrays of integers—must be converted to tensors before fed into
 the neural network. This conversion can be done a couple of ways:

1 Convert the arrays into vectors of 0s and 1s indicating word occurrence, 
similar to a one-hot encoding. For example, the sequence [3, 5] would become a 
10,000-dimensional vector that is all zeros except for indices 3 and 5, which 
are ones. Then, make this the first layer in our network—a Dense layer—that can
handle floating point vector data. This approach is memory intensive, though, 
requiring a num_words * num_reviews size matrix.
 
--2 Alternatively, we can pad the arrays so they all have the same length, ----
then create an integer tensor of shape max_length * num_reviews. We can use an 
embedding layer capable of handling this shape as the first layer in our 
network.

In this tutorial, we will use the second approach.
Since the movie reviews must be the same length, we will use the pad_sequences 
function to standardize the lengths:
"""

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# now all the train_data_len == 256
# print(train_data[0])  # the tail are all zero
# Build the model
vocab_size = 10000

"""
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = 'relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))
"""

model = keras.Sequential([
        keras.layers.Embedding(vocab_size,16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation = 'relu'),
        keras.layers.Dense(1, activation = 'sigmoid')          
        ])
model.summary()  

model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

# Create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# history is a History object that contains everything happended during training
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val),
                    verbose = 1)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)

# Create a graph of acc and loss over time
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()































