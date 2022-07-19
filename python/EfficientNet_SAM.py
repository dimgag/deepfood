# Author: Dimitrios Gagatsis
# Date: 2022-07-18
# Description: EfficientNet-SAM model
from email.mime import base
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
train_data_dir = '/Users/dim__gag/python/food-101/data_mini/train_mini'
validation_data_dir = '/Users/dim__gag/python/food-101/data_mini/test_mini'
n_classes = 3
nb_train_samples = 2250
nb_validation_samples = 750
# Training configuration
img_width, img_height = 299, 299
batch_size = 16
epochs = 30

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
try: # detect TPUs
    tpu = None
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
print("Number of accelerators: ", strategy.num_replicas_in_sync)

# <-------------------------------------------------------------------------------------------------------------->
# GET THE EFFICIENTNET MODEL

from efficientnet_v2 import EfficientNetV2S

base_EffNet = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=None, input_shape= (299,299,3), pooling=None, classes=101, classifier_activation="softmax")
base_EffNet.trainable = False # Freeze the model
# Fine tune the model
x = base_EffNet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(101, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)
EffNet = Model(inputs=base_EffNet.input, outputs=predictions)
EffNet.trainable = True # Unfreeze the model



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

# GET THE SAM MODEL
# Initialize the model
with strategy.scope():
    model = SAMModel(EffNet)



# <-------------------------------------------------------------------------------------------------------------->
# Compile the model
# model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='EfficientSAM.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('EfficientSAM.log')

# Treain the model
history = model.fit_generator(train_generator, steps_per_epoch = nb_train_samples // batch_size, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size, epochs=epochs, verbose=1, callbacks=[csv_logger, checkpointer])

history = model.fit(train_generator, steps_per_epoch = nb_train_samples // batch_size, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size, epochs=epochs, verbose=1, callbacks=[csv_logger, checkpointer])


# EffNet Only 
EffNet.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), metrics=['accuracy'])
history = EffNet.fit_generator(train_generator, steps_per_epoch = nb_train_samples // batch_size, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size, epochs=epochs, verbose=1, callbacks=[csv_logger, checkpointer]) 


model.save('EfficientSAM.hdf5')


# <-------------------------------------------------------------------------------------------------------------->
# apply the sam optimizer to the model
import sam as SAM

opt = tf.keras.optimizers.SGD(learning_rate = 0.0001)
opt = SAM.SAMWarpper(opt, rho=0.05)


inputs = base_EffNet.input
labels = base_EffNet.output

def loss_func(predictions, labels):
    return tf.keras.losses.categorical_crossentropy(predictions, labels)

model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def grad_func():
    with tf.GradientTape() as tape:
        pred = model(inputs, training=True)
        loss = loss_func(predictions=pred, labels=labels)
    return pred, loss, tape


opt.optimize(grad_func, model.trainable_variables)


# <-------------------------------------------------------------------------------------------------------------->
# EffNet Only 
EffNet.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), metrics=['accuracy'])
history = EffNet.fit_generator(train_generator, steps_per_epoch = nb_train_samples // batch_size, validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size, epochs=epochs, verbose=1, callbacks=[csv_logger, checkpointer]) 

