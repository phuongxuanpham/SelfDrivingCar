#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.backend import tf as ktf
from keras.callbacks import ModelCheckpoint

def flip_image(img, angle):
    """
    Randomly flip the image and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        angle = -angle
    return img, angle
    
def select_image(batch_sample, is_training=False):
    """
    Randomly select an image among the center, left or right images, and adjust the steering angle.
    This way, we can teach your model how to steer if the car drifts off to the left or the right.
    """
    if is_training == True:
        choice = np.random.choice(3)
    else:
        choice = 0
        
    name = '../../P3_Data/IMG/'+batch_sample[choice].split('/')[-1]
    image = cv2.imread(name)
    steering_center = float(batch_sample[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    if choice == 0:
        return image, steering_center
    elif choice == 1:
        return image, steering_left
    return image, steering_right

def generator(samples, batch_size=32, is_training=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image, angle = select_image(batch_sample, is_training=is_training)
                images.append(image)
                angles.append(angle)

            # Get training data
            X_train = np.array(images)
            y_train = np.array(angles)
            
            # Randomly flip image if in training mode
            if is_training == True:
                X_train_augmented, y_train_augmented = [], []
                for x, y in zip(X_train, y_train):
                    x_augmented, y_augmented = flip_image(x, y)
                    X_train_augmented.append(x_augmented)
                    y_train_augmented.append(y_augmented)

                X_train_augmented = np.array(X_train_augmented)
                y_train_augmented = np.array(y_train_augmented)       

                yield sklearn.utils.shuffle(X_train_augmented, y_train_augmented)
            
            else:
                yield sklearn.utils.shuffle(X_train, y_train)

def build_network(resizing=False):
    # Build network architecture 
    # for a regression network (need only 1 neuron at output)  
    row, col, ch = 160, 320, 3  # image format
    input_shape = (row,col,ch)
    
    def resize(image):
        from keras.backend import tf as ktf   
        resized = ktf.image.resize_images(image, (66, 200))
        return resized
    
    # Create the Sequential model
    model = Sequential()
    
    ## Set up lambda layers for data preprocessing: 
    
    # Set up cropping2D layer: cropping (top, bottom) (left, right) pixels 
    model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=input_shape)) 
    
    # Add Lambda layer for resizing image (image, height, width, data_format)
    if resizing == True:
        model.add(Lambda(resize, input_shape=(75, 320, 3), output_shape=(66, 200, 3)))
    
    # Add Lambda layer for normalization
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))
    
    ## Build a Multi-layer feedforward neural network with Keras here.
    
    # 1st Layer - Add a convolution layer
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    
    # 2nd Layer - Add a convolution layer
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    
    # 3rd Layer - Add a convolution layer
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    
    # 4th Layer - Add a convolution layer
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    # 5th Layer - Add a convolution layer
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    # 6th Layer - Add a convolution layer
    model.add(Dropout(0.5))
    
    # 7th Layer - Add a flatten layer
    model.add(Flatten())
    
    # 8th Layer - Add a fully connected layer
    model.add(Dense(100, activation='relu'))
    
    # 9th Layer - Add a fully connected layer
    model.add(Dense(50, activation='relu'))
    
    # 10th Layer - Add a fully connected layer
    model.add(Dense(10, activation='relu'))
    
    # 11th Layer - Add a fully connected layer
    model.add(Dense(1))
    
    model.summary()   

    return model             

def train_network(model, train_generator, train_samples, validation_generator, validation_samples, nb_epoch=10):
    # saves the model weights after each epoch if the validation loss decreased
    checkpointer = ModelCheckpoint('model-{epoch:02d}.h5',
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=False,
                                     mode='auto')
    # Compile and train the model
    model.compile(optimizer='adam', loss='mse', verbose = 1)
    # history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, batch_size=128)
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                                         validation_data=validation_generator, nb_val_samples=len(validation_samples), \
                                         nb_epoch=nb_epoch, callbacks=[checkpointer], verbose=1)
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    
def main(): 
    
    # read data
    samples = []
    with open('../../P3_Data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    
    # split data to train and validation sets
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=0)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32, is_training=True)
    validation_generator = generator(validation_samples, batch_size=32, is_training=False)
    
    # Build network 
    model = build_network(resizing=False)
    
    # Train the network
    train_network(model, train_generator, train_samples, validation_generator, validation_samples, nb_epoch=10)
    
if __name__ == '__main__':
    main()