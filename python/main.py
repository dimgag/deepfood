from types import new_class
import models as m
import train as t
import visualization as viz
import evaluate as ev
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Full Food-101 dataset
train_data_dir = 'persistent/food-101/train'
validation_data_dir = 'persistent/food-101/test'
n_classes = 101
nb_train_samples = 75750
nb_validation_samples = 25250

# # Small Food-101 dataset
# train_data_dir = '/Users/dim__gag/python/food-101/data_mini/train_mini'
# validation_data_dir = '/Users/dim__gag/python/food-101/data_mini/test_mini'
# n_classes = 3
# nb_train_samples = 2250
# nb_validation_samples = 750


# Training configuration
img_width, img_height = 299, 299 # input image dimensions
batch_size = 16 # Change this to your desired batch size
epochs = 300 # Change the Number of epochs here

# Note: You need to change the number of classes in the model.py (line: 32)

# Data Augementation
train_generator, validation_generator = m.data_augmentation(train_data_dir=train_data_dir, validation_data_dir=validation_data_dir, img_width=img_height, img_height=img_height, batch_size=batch_size)

# Load the models
models = ['EfficientNetV2S'] # ['ResNet152V2', 'InceptionV3', 'VGG16', 'Xception', 'EfficientNetV2S']

for mod in models:
    model = m.get_model(mod, False, 'imagenet')
    # Fine Tuning
    model = m.model_finetuning(model)
    # Model Training
    model, model_history = t.train_model(model=model, model_name=mod, train=train_generator, val=validation_generator, nb_train_samples=nb_train_samples, nb_validation_samples=nb_validation_samples, epochs=epochs, batch_size=batch_size)
    # # Visualization
    viz.plot_Acc_and_Loss(model_history, title='Accuracy and Loss of the model')
    # # Evaluate
    ev.model_eval(model, train_generator, validation_generator)