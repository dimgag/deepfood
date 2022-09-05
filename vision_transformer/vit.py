'''
Implementation of the ViT model in Keras for Food Classification on the Food-101 dataset.
In this implementation, the model is trained on the Food-101 dataset and evaluated on the Food-101 test set.
Still in progress.
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Source: https://keras.io/examples/vision/image_classification_with_vision_transformer/
# # Full Food-101 dataset
# !wget https://s3.amazonaws.com/fast-ai-imageclas/food-101.tar.gz
# !tar -xvzf food-101.tar.gz
train_data_dir = '/Users/dim__gag/Desktop/food-101/train'
validation_data_dir = '/Users/dim__gag/Desktop/food-101/test'
# Data parameters
num_classes = 101
input_shape = (299, 299, 3)
img_width, img_height = 299, 299 # input image dimensions
batch_size = 32 # Change this to your desired batch size
epochs = 100 # Change the Number of epochs here

# Data Augmentation
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



train_generator, validation_generator = data_augmentation(train_data_dir=train_data_dir, validation_data_dir=validation_data_dir, img_width=img_height, img_height=img_height, batch_size=batch_size)

# Number of class indices
print(train_generator.class_indices)


# Build the ViT model

# Hypererparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 299  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Implement patch creation as a layer.
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    # call for food-101 dataset
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches



# Visualize the patches of a single image
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
image = train_generator[0][0][0]
plt.imshow(image)
plt.axis("off")
plt.show()


resized_image = tf.image.resize(tf.convert_to_tensor([image]), size=(image_size, image_size))


# Create the patch layer.
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy())
    plt.axis("off")