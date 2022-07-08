from collections import defaultdict
import collections
import shutil

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


# Test Data
classes_images=defaultdict(list)
with open('../persistent/food-101/meta/test.txt', 'r') as txt:
	paths= [read.strip() for read in txt.readlines()]
	for p in paths:
		food = p.split('/')
		classes_images[food[0]].append(food[1] + '.jpg')