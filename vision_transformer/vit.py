'''
Implementation of the ViT model in Keras for Food Classification on the Food-101 dataset.
In this implementation, the model is trained on the Food-101 dataset and evaluated on the Food-101 test set.
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
epochs = 1 # Change the Number of epochs here

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
# print(train_generator.class_indices)


# Build the ViT model

# Hypererparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
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

# Implement patch creation layer
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



"""
# Visualize the patches of a single image
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
image = train_generator[0][0][0]
plt.imshow(image)
plt.axis("off")
resized_image = tf.image.resize(tf.convert_to_tensor([image]), size=(image_size, image_size))
# Create the patch layer.
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")
# Visualize the patches
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy())
    plt.axis("off")
plt.show()
"""


# Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


"""
Test the patch encoding layer:
encoded_patches_1 = PatchEncoder(num_patches, projection_dim)(patches)
encoded_patches_1.shape
print(encoded_patches_1)"""

# Create ViT model
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# Create the ViT classifier
vit_classifier = create_vit_classifier()
# vit_classifier.summary()


# Compile the model
vit_classifier.compile(
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

checkpoint_filepath = "~/vision_transformer/food-101-vit.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True)


# Train the model
history = vit_classifier.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[model_checkpoint_callback])


# Evaluate the model
test_loss, test_acc = vit_classifier.evaluate(test_generator, verbose=2)
print('Validation loss:', test_loss)
print('Validation accuracy:', test_acc)