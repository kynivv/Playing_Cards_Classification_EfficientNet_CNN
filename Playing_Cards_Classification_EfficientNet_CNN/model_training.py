import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

from tensorflow import keras
from keras import layers
from zipfile import ZipFile
from keras.callbacks import ModelCheckpoint
from glob import glob
from sklearn.model_selection import train_test_split


## Data Extraction
#with ZipFile('playing_cards.zip') as playing_cards:
#    playing_cards.extractall('dataset')


# Constants & Hyperparameters
SPLIT = 0.25

BATCH_SIZE = 5
EPOCHS = 10

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


# Data Preprocessing
X = []
Y = []

train_path = 'dataset/train'
test_path = 'dataset/test'

classes = os.listdir(train_path)

for i, name in enumerate(classes):
    images = glob(f'{train_path}/{name}/*.jpg')

    for image in images:
        img = cv2.imread(image)
        X.append(img)
        Y.append(i)

X = np.asarray(X)
Y = pd.get_dummies(Y)


# Train Test Data Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    shuffle= True,
                                                    random_state= 24,
                                                    test_size= SPLIT
                                                    )


# Creating Model Based on EfficientNet
base_model = keras.applications.EfficientNetB3(include_top= False, 
                                               input_shape= IMG_SHAPE,
                                               pooling= 'max',
                                               weights= 'imagenet'
                                               )

model = keras.Sequential([
    base_model,

    layers.Dense(53, activation= 'softmax')
])

model.compile(optimizer= 'adam',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy']
              )


# Model Callbacks
checkpoint = ModelCheckpoint('output/finalmodel.h5',
                             monitor= 'val_accuracy',
                             verbose= 1,
                             save_best_only= True,
                             save_weights_only= False
                             )


# Model Training
model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          verbose= 1,
          validation_data= (X_test, Y_test),
          callbacks= checkpoint
          )