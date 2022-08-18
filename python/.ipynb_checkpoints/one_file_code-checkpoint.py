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
# from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, EfficientNetV2L
# !pip install tf-nightly
from tensorflow.python.keras.applications.efficientnet import *
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()


# # # # # # # # # 

# Configuration
n_classes = 101
img_width, img_height = 224, 224
train_data_dir = 'persistent/food-101/train'
validation_data_dir = 'persistent/food-101/test'
nb_train_samples = 75750
nb_validation_samples = 25250
batch_size = 16
epochs = 30

# Data
def data_augmentation(train_data_dir, validation_data_dir, img_height, img_width, batch_size):
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    return train_generator, validation_generator



# # # # # # # # # # # # # # # # # # # # # # # # # # 
# Plot Accuracy and Loss of the model
def plot_Acc_and_Loss(history,title):
    # plot model accuracy
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title(title)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    # plot model loss
    plt.subplot(1,2,2)
    plt.title(title)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()

# Print training, test accuracy and loss of the model
def model_eval(model, train, val):
    # evaluate the model
    train_loss, train_acc = model.evaluate(train, verbose=0)
    val_loss, val_acc = model.evaluate(val, verbose=0)
    print('Train loss:', train_loss)
    print('Train accuracy:', train_acc)
    print('Validation loss:', val_loss)
    print('Validation accuracy:', val_acc)
# # # # # # # # # # # # # # # # # # # # # # # # # # 



# Data Augementation
train_generator, validation_generator = data_augmentation(train_data_dir=train_data_dir, validation_data_dir=validation_data_dir, img_width=img_height, img_height=img_height, batch_size=batch_size)

# Load the models
# models = ['ResNet152V2', 'InceptionV3', 'VGG16', 'Xception', 'EfficientNetV2S']

model =  tf.keras.applications.ResNet152V2(include_top=None, weights=None, input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=102, classifier_activation="softmax")
model.trainable = False

# Fine Tune the model
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(101, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)



# Train the model
model.trainable = True
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='ResNet152V2_food101.hdf5', verbose=1, save_best_only=True)

csv_logger = CSVLogger('history_ResNet152V2_food101.hdf5')

# history = model.fit_generator(train_generator, steps_per_epoch = nb_train_samples // batch_size, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size, epochs=epochs, verbose=1, callbacks=[csv_logger, checkpointer])


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)


# model.save('ResNet152V2_food101.hdf5')


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("/workspace/persistent/deepfood/ResNet152V2_food101.hdf5", "wb") as f:
    f.write(tflite_model)


plot_Acc_and_Loss(model_history, title='Accuracy and Loss of the model')

model_eval(model, train_generator, validation_generator)
