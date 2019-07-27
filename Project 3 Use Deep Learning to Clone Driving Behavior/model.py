#All Imports
import os
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

#Collecting samples from the driving_log.csv file and spliting it into training and validation samples
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)

#Generator function for generating data for training and validation
def generator(samples, batch_size=32):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    image_name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(image_name)
                    flipped_image = np.fliplr(image)
                    angle = float(batch_sample[3])
                    if(i==1):
                        angle += 0.25
                    elif(i==2):
                        angle -= 0.25
                    angle_flipped = -angle
                    images.append(image)
                    images.append(flipped_image)
                    angles.append(angle)
                    angles.append(angle_flipped)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Based on NVIDIA model architecture
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Cropping the image to get the required region of image
model.add(Cropping2D(cropping=((67,25), (0,0))))
model.add(Convolution2D(24, 5, 5))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3))
model.add(LeakyReLU())
model.add(Convolution2D(64, 3, 3))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(LeakyReLU())
model.add(Dense(50))
model.add(LeakyReLU())
model.add(Dense(10))
model.add(LeakyReLU())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=3000, nb_epoch=10)

# Saving the model
model.save('model.h5')