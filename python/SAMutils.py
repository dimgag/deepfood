# import resnet_cifar10 # This is the net that is isung.. I want to replace this with a custom net... but I'm not sure how to do that...
# from pyexpat import model
import tensorflow as tf
import matplotlib.pyplot as plt

'''# Reference 
# https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/blob/master/zoo/resnet/resnet_cifar10.py
def get_training_model():
    # ResNet20
    n = 2
    depth =  n * 9 + 2
    n_blocks = ((depth - 2) // 9) - 1

    # The input tensor
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    # The Stem Convolution Group
    x = resnet_cifar10.stem(inputs)

    # The learner
    x = resnet_cifar10.learner(x, n_blocks)

    # The Classifier for 10 classes
    outputs = resnet_cifar10.classifier(x, 10)

    # Instantiate the Model
    model = tf.keras.Model(inputs, outputs)
    
    return model

def plot_history(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
'''
# Custom Function
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
# from tensorflow.keras import regularizers
# from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras import regularizers

def get_training_model():
    from efficientnet_v2 import EfficientNetV2S
    base_EffNet = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=102, classifier_activation="softmax")
    base_EffNet.trainable = False # Freeze the model
    # Fine tune the model
    x = base_EffNet.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(101, activation='softmax')(x)
    model = Model(inputs=base_EffNet.input, outputs=predictions)
    model.trainable = True # Unfreeze the model
    return model






# BACKUP
# def get_training_model():
#     from efficientnet_v2 import EfficientNetV2S
#     base_EffNet = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=101, classifier_activation="softmax")
#     base_EffNet.trainable = False # Freeze the model
#     # Fine tune the model
#     x = base_EffNet.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(101, kernel_regularizer=l2(0.005), activation='softmax')(x)
#     model = Model(inputs=base_EffNet.input, outputs=predictions)
#     model.trainable = True # Unfreeze the model
#     return model
