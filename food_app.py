# Food Image Recognition Application 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
# %matplotlib inline
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load model
model_best = load_model('/Users/dim__gag/Desktop/EfficientNetV2L/EfficientNetV2L.hdf5', compile=False)

# Load food labels
data_dir = "/Users/dim__gag/Desktop/food-101/images" # This is for local path
foods_sorted = sorted(os.listdir(data_dir))

def pick_n_random_classes(n):
  food_list = []
  random_food_indices = random.sample(range(len(foods_sorted)),n) # We are picking n random food classes
  for i in random_food_indices:
    food_list.append(foods_sorted[i])
  food_list.sort()
  return food_list


def predict_class(model, images, show = True):
  for img in images:
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value) 
        plt.show()



food_list = pick_n_random_classes(101)

images = []

# image = input("Give the image path:")

# images.append('/Users/dim__gag/git/deepfood/test_images/steak.jpg')

images.append(input("Give the image path:"))

# print(images)

predict_class(model_best, images, show = True)


