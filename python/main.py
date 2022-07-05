import models as m
import train as t
import visualization as viz
import evaluate as ev

# Configuration
n_classes = 101
img_width, img_height = 299, 299
train_data_dir = 'train'
validation_data_dir = 'test'
nb_train_samples = 75750
nb_validation_samples = 25250
batch_size = 16
epochs = 30


# Data Augementation
train_datagen, validation_generator = t.data_augmentation(train_data_dir, validation_data_dir, img_width, img_height, batch_size)


# Load the models
models = ['ResNet152V2', 'InceptionV3', 'VGG16', 'Xception', 'EfficientNetV2S']

for model in models:
    model = m.get_model(model, False, 'imagenet')
    # Fine Tuning
    model = m.model_finetuning(model)
    # Import Training
    model, model_history = t.train_model(model, 'inceptionV2_food101', train_generator, validation_generator, nb_train_samples, nb_validation_samples, epochs, batch_size)
    # Visualization
    viz(model_history, title='Accuracy and Loss of the model')
    # Evaluate
    ev(model, train_datagen, validation_generator)
    
