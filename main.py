# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 01:17:55 2021

@author: Kiran
"""

# This project is done using 'A Large Scale Fish Dataset' created by Oğuzhan Ulucan available on Kaggle at https://www.kaggle.com/crowww/a-large-scale-fish-dataset.
# It was run on Kaggle notebook, so data was directly  imported from kaggle, please make sure to change the data directory if you're runnning the code on your machine.



#%% Importing stuff
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#%% Reading the data

main_directory = os.listdir("../input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset")
directory = ("../input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset")

# Reading the images using keras
data_reader = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    subset='validation',
    seed=200,
    validation_split=0.2,
    labels="inferred"
    )

# Printing the names of the classes
class_names  = data_reader.class_names
print(class_names)

# Printing an image from the dataset

plt.figure(figsize=(10, 10))
for images, labels in data_reader.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
#%% shape and batch of images

for image_batch, labels_batch in data_reader:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

#%% Standardising the images

# # shape and batch of images
# for image_batch, labels_batch in data_reader:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

# I am not standardizing the pixel values now because I am using keras to rescale the pixel values in the model.

#%% Creating a cnn network
# first layer of the model will rescale the images

cnn = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,4,activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(9)
    ])

cnn.compile(optimizer = 'adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy'])


cnn_history = cnn.fit(data_reader, epochs=10)

#%% plotting the training

loss = cnn_history.history['loss']
accuracy = cnn_history.history['accuracy']

plt.plot(loss, 'r', label='loss')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc=0)
plt.figure()
plt.show()


plt.plot(accuracy, 'g', label='accuracy')
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc=0)
plt.figure()
plt.show()

#%% testing the prediction using a new image

import keras
img_height = 256
img_width = 256
sunflower_path = "../input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset/Black Sea Sprat/Black Sea Sprat/00001.png"
#sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = cnn.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
