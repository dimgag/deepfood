# 🍕 DeepFood 🍝
## 🥖 Multiclass Classification using Tensorflow on Food-101 Dataset 🥐
-----------------------------------------------------------------------------------------
<p align="left"> <img src="https://komarev.com/ghpvc/?username=dimgag&label=Profile%20views&color=0e75b6&style=flat-square" alt="dimgag/deepfood" /> </p>

### 🍟 Download & Extract Food-101 Dataset 🍔
```
!wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
!tar xzvf food-101.tar.gz
```
To split the dataset into Train and Test run the DSRI_split_train_test.py in your terminal after selecting the desired directory.

-----------------------------------------------------------------------------------------
### 🌯 Install requirements 🌮
```
pip install -r requirements.txt
```
-----------------------------------------------------------------------------------------
### 🍪 Files Overview 🥛
```
food-101 > images - Format of the Food-101 dataset and how to be splitted into Train and Test
         > meta
         > test
         > train 

models   > EfficientNetV2L          > assets
                                    > variables 
                                    > EfficiencyNetV2L.hdf5
                                    > EfficiencyNetV2L.log
                                    > kears_metadata.pb
                                    > saved_model.pb

         > EfficientNetV2S          > assets
                                    > variables 
                                    > EfficiencyNetV2S.hdf5
                                    > EfficiencyNetV2S.log
                                    > kears_metadata.pb
                                    > saved_model.pb

         > EfficientNetV2S_25Epochs > EfficiencyNetV2S.hdf5
                                    > EfficiencyNetV2S.log

python   > evaluate.py - Evaluate the model on the test set
         > main.py - Main script to run the model
         > models.py - Models definition + Fine Tuning
         > split_train_test.py - Create the data folders in DSRI persistent folder
         > train.py - Train the model
         > visualization.py - Visualize the model output

test_images            - images used for testing the model
DeepFood_Food101.ipynb - Code Notebook with Models and Data
README.md              - README
requirements.txt       - Requirements for the project
```


### Citation for food-101 dataset:
```
@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
```
