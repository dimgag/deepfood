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
from efficientnet_v2 import EfficientNetV2S, EfficientNetV2L
# from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, EfficientNetV2L
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D



def get_model(model_name, include_top, weights):
    if model_name == 'ResNet152V2':
        model =  tf.keras.applications.ResNet152V2(include_top=include_top, weights=weights, input_tensor=None, input_shape=None, pooling=None, classes=101, classifier_activation="softmax")
        model.trainable = False
    elif model_name == 'InceptionV3':
        model =  tf.keras.applications.InceptionV3(include_top=include_top, weights=weights, input_tensor=None, input_shape=None, pooling=None, classes=101, classifier_activation="softmax")
        model.trainable = False
    elif model_name == 'VGG16':
        model =  tf.keras.applications.VGG16(include_top=include_top, weights=weights, input_tensor=None, input_shape=None, pooling=None, classes=101, classifier_activation="softmax")
        model.trainable = False
    elif model_name == 'Xception':
        model =  tf.keras.applications.Xception(include_top=include_top, weights=weights, input_tensor=None, input_shape=None, pooling=None, classes=101, classifier_activation="softmax")
        model.trainable = False
    elif model_name == 'EfficientNetV2S':
        model = EfficientNetV2S(include_top=include_top, weights=weights, input_tensor=None, input_shape=None, pooling=None, classes=3, classifier_activation="softmax")
        model.trainable = False
    elif model_name == 'EfficientNetV2L':
        model = EfficientNetV2L(include_top=include_top, weights=weights, input_tensor=None, input_shape=None, pooling=None, classes=3, classifier_activation="softmax")
        model.trainable = False
    return model

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


def model_finetuning(model):
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(101,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    return model