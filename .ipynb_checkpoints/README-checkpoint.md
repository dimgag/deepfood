# 🍕 DeepFood 🍔
## Multiclass Classification using Keras and Tensorflow on Food-101 Dataset
-----------------------------------------------------------------------------------------
### Download & Extract Food-101 Dataset
```
!wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
!tar xzvf food-101.tar.gz
```
To split the dataset into Train and Test run the DSRI_split_train_test.py in your terminal after selecting the desired directory.

-----------------------------------------------------------------------------------------
### Create conda env with the requirements
```bash
chmod +x install.sh

./install.sh
```
-----------------------------------------------------------------------------------------
## Files
```
python  > DSRI_split_train_test.py - Create the data folders in DSRI persistent folder
        > evaluate.py
        > main.py
        > models.py
        > train.py
        > visualization.py

DeepFood_Food101.ipynb - Code Notebook with Models and Data

env.yml

install.sh
readme_steps.txt - Steps in DSRI Terminal
README.md
requirements.txt - pip Requirements
```

-----------------------------------------------------------------------------------------
## Models
* ResNet152V2
* Inceptionv3
* VGG16
* Xception
* EfficientNetV2L , EfficientNetV2S