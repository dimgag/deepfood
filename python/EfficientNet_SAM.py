# Author: Dimitrios Gagatsis
# Date: 2022-07-18
# Description: EfficientNet-SAM model
import os
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
import SAMModel
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# <-------------------------------------------------------------------------------------------------------------->
# # Small Food-101 dataset
# train_data_dir = '/Users/dim__gag/python/food-101/data_mini/train_mini'
# validation_data_dir = '/Users/dim__gag/python/food-101/data_mini/test_mini'
# n_classes = 3
# nb_train_samples = 2250
# nb_validation_samples = 750

# # Full Food-101 dataset
train_data_dir = 'persistent/food-101/train'
validation_data_dir = 'persistent/food-101/test'
n_classes = 101
nb_train_samples = 75750
nb_validation_samples = 25250

# Training configuration
img_width, img_height = 299, 299
batch_size = 16 # Change to 32?
epochs = 100

# <-------------------------------------------------------------------------------------------------------------->
# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


# <-------------------------------------------------------------------------------------------------------------->
# Define strategy for the training of the models
# try: # detect TPUs
#     tpu = None
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.experimental.TPUStrategy(tpu)
# except ValueError: # detect GPUs
#     strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
# print("Number of accelerators: ", strategy.num_replicas_in_sync)

# <-------------------------------------------------------------------------------------------------------------->
# GET THE EFFICIENTNET MODEL

# from efficientnet_v2 import EfficientNetV2S

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, EfficientNetV2L

base_EffNet = EfficientNetV2S(include_top=False, weights='imagenet')
base_EffNet.trainable = False # Freeze the model
# Fine tune the model
x = base_EffNet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(101, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)
EffNet = Model(inputs=base_EffNet.input, outputs=predictions)
# EffNet.trainable = True # Unfreeze the model
# EffNet.summary()



# <-------------------------------------------------------------------------------------------------------------->
class SAMModel(tf.keras.Model):
    def __init__(self, model, rho=0.05):
        """
        p, q = 2 for optimal results as suggested in the paper
        (Section 2)
        """
        super(SAMModel, self).__init__()
        self.model = model
        self.rho = rho

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.compiled_loss(labels, predictions)    
        
        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)
        
        self.optimizer.apply_gradients(
            zip(sam_gradients, trainable_params))
        
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self.model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm
# <-------------------------------------------------------------------------------------------------------------->

'''# GET THE SAM MODEL
# Initialize the model
# with strategy.scope():
model = SAMModel(EffNet)
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# checkpointer = ModelCheckpoint(filepath='EfficientSAM.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('EfficientSAM.log')
# Train the model
history = model.fit(train_generator, steps_per_epoch = nb_train_samples // batch_size, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size, epochs=epochs, verbose=1, callbacks=[csv_logger])

# model.save('EfficientSAM.hdf5')'''





# <-------------------------------------------------------------------------------------------------------------->

model = SAMModel(EffNet)
# model = EffNet


# GET THE SAM MODEL
model.trainable = True # Unfreeze the model
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# checkpointer = ModelCheckpoint(filepath='EfficientSAM.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('EfficientSAM_1.log')

history = model.fit(train_generator, steps_per_epoch = nb_train_samples // batch_size, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size, epochs=100, verbose=1, callbacks=[csv_logger]) 


# plot_Acc_and_Loss(history, "EfficientNetV2S + SAM")