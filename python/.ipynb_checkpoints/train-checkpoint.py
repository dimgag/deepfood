import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import cv2
import random
# %matplotlib inline
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, EfficientNetV2L
# from tensorflow.python.keras.applications.efficientnet import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D



def train_model(model, model_name, train, val, nb_train_samples, nb_validation_samples, epochs, batch_size):
    model.trainable = True
    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath=model_name+'.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(model_name +'.log')
    history = model.fit_generator(train,
                                  steps_per_epoch = nb_train_samples // batch_size,
                                  validation_data=val,
                                  validation_steps=nb_validation_samples // batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[csv_logger, checkpointer])
    model.save(model_name)
    return model, history