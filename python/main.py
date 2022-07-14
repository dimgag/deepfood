import models as m
import train as t
import visualization as viz
import evaluate as ev

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
n_classes = 101
img_width, img_height = 299, 299
train_data_dir = 'persistent/food-101/train'
validation_data_dir = 'persistent/food-101/test'
nb_train_samples = 75750
nb_validation_samples = 25250
batch_size = 16
epochs = 30


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




# EfficientNet = m.get_model('EfficientNetV2S', False, 'imagenet')
# EfficientNet.summary()

# EfficientNet = m.model_finetuning(EfficientNet)
# EfficientNet.summary()