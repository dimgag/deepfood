# 🍕 DeepFood 🍝
## 🥖 Multiclass Classification using Tensorflow on Food-101 Dataset 🥐

## 🍟 Download & Extract Food-101 Dataset 🍔
```
!wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
!tar xzvf food-101.tar.gz
```
To split the dataset into Train and Test run the python/split_train_test.py in your terminal after selecting the desired directory.

<!-- empty space -->
<br><br>


## 🌯 Install requirements 🌮
```
pip install -r requirements.txt
```
<!-- empty space -->
<br><br>


## 🍪 Files Overview 🥛
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
readme_images               - Images used in the README.md
test_images                 - images used for testing the model

vision_transformer     > vit_cifar100.py - Vision Transformer model definition
                       > vit.py - Vision Transformer model definition

DeepFood_Food101.ipynb          - Code Notebook with Models and Data
EfficientNetV2_Evaluation.ipynb - Code Notebook with Models and Data Evaluation
logs_analysis.ipynb             - Code Notebook with Logs Analysis
model_predictions.ipynb         - Code Notebook with Model Predictions

README.md                       - README
requirements.txt                - Requirements for the repository
```

<!-- empty space -->
<br><br>


# ⚙️ Training Configuration ⚙️

|                                |                                 |
| :----------------------------: | :-----------------------------: |
|Number of Classes               | 101                             |
|Number of training samples      | 75750                           |
|Number of validation samples    | 25250                           |
|Input image dimensions          | (299, 299)                      |
|Batch size                      | 32                              |
|Number of Epochs                | 100                             |
|Learning Rate                   | 0.0001                          |
|Momentum                        | 0.9                             |
|Optimizer                       | Stochastic gradient descent     |
|Loss Function                   | Categorical crossentropy        |
|Evaluation Metric               | Accuracy                        |
|GPU                             | NVIDIA Tesla V100 SXM2 32 GB    |


<!-- empty space -->
<br><br>


# 🚀 Models Results 
<!-- models table -->
| Model | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss | # Parameters | # Layers |
| :---: | :---------------: | :-----------------: | :-----------: | :-------------: | :----------: | :------: |
| [EfficientNetV2-S](https://github.com/dimgag/deepfood/blob/master/models/EfficientNetV2S) | 0.9129 | 0.8337 | 0.4350 | 0.7551 | 20.5M | 516 |  
| [EfficientNetV2-L](https://github.com/dimgag/deepfood/blob/master/models/EfficientNetV2L) | 0.9411 | 0.8463 | 0.3394 | 0.7650 | 117.9M | 1031 |


<!-- empty space -->
<br><br>


# 📊 Visualization EfficientNetV2-S vs EfficientNetV2-L 📈
<!-- visualization table -->
<!-- add Figure 1 -->

<img align="center" src="https://github.com/dimgag/deepfood/blob/master/readme_images/EffNetS_vs_EffNetL.png" alt="EffNetS_vs_EffNetL" width="727" height="704">


<!-- empty space -->
<br><br>


# 🍽 Predictions
<img align="center" src="https://github.com/dimgag/deepfood/blob/master/readme_images/Predictions.png" alt="EffNetS_vs_EffNetL" width="728" height="490">

<!-- empty space -->
<br><br>

# 🚀 Test the model
```
# Activate your python env with the requirement.txt
# Then Run 
python food_app.py
```

<!-- empty space -->
<br><br>

# Future code to do
- ViT for food recognition task
- Test models on images with added noise (filters, rotations, etc). 

<!-- empty space -->
<br><br>

# 🍺 Acknowledgements 🍻
- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [EfficientNetV2](https://arxiv.org/abs/2104.00298)
