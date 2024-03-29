from collections import defaultdict
import shutil
import os

# Split the dataset into train and test -> RUN THIS IN TERMINAL
# Train Data
classes_images=defaultdict(list)
with open('../persistent/food-101/meta/train.txt', 'r') as txt:
	paths= [read.strip() for read in txt.readlines()]
	for p in paths:
		food = p.split('/')
		classes_images[food[0]].append(food[1] + '.jpg')

for food in classes_images.keys():
	if not os.path.exists(os.path.join("../persistent/food-101/train",food)):
		os.makedirs(os.path.join("../persistent/food-101/train", food))
	for i in classes_images[food]:
		shutil.copyfile(os.path.join("../persistent/food-101/images", food, i), os.path.join("../persistent/food-101/train", food, i))
  
# Test/Validation Data
classes_images=defaultdict(list)
with open('../persistent/food-101/meta/test.txt', 'r') as txt:
	paths= [read.strip() for read in txt.readlines()]
	for p in paths:
		food = p.split('/')
		classes_images[food[0]].append(food[1] + '.jpg')







'''from collections import defaultdict
import collections
import shutil
import os

# Train Data
classes_images=defaultdict(list)
with open('/Users/dim__gag/Desktop/food-101/meta/train.txt', 'r') as txt:
	paths= [read.strip() for read in txt.readlines()]
	for p in paths:
		food = p.split('/')
		classes_images[food[0]].append(food[1] + '.jpg')

for food in classes_images.keys():
	if not os.path.exists(os.path.join("/Users/dim__gag/Desktop/food-101/train",food)):
		os.makedirs(os.path.join("/Users/dim__gag/Desktop/food-101/train", food))
	for i in classes_images[food]:
		shutil.copyfile(os.path.join("/Users/dim__gag/Desktop/food-101/images", food, i), os.path.join("/Users/dim__gag/Desktop/food-101/train", food, i))
  '''





'''# Test/Validation Data
classes_images=defaultdict(list)
with open('/Users/dim__gag/Desktop/food-101/meta/test.txt', 'r') as txt:
	paths= [read.strip() for read in txt.readlines()]
	for p in paths:
		food = p.split('/')
		classes_images[food[0]].append(food[1] + '.jpg')

for food in classes_images.keys():
	if not os.path.exists(os.path.join("/Users/dim__gag/Desktop/food-101/test",food)):
		os.makedirs(os.path.join("/Users/dim__gag/Desktop/food-101/test", food))
	for i in classes_images[food]:
		shutil.copyfile(os.path.join("/Users/dim__gag/Desktop/food-101/images", food, i), os.path.join("/Users/dim__gag/Desktop/food-101/test", food, i))'''