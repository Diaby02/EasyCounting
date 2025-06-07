# Class-Agnostic Object Counting Model for manufacturing applications

Author: Nicolas Bourez

## Introduction

This code aims to compare different class-agnostic counting models in order to understand their capabilities
when applied in industrial data. For this purpose, the code from these different models might have been included in this repository.

The link to the official code of the different models are listed below:

* [FamNet+](https://github.com/cvlab-stonybrook/LearningToCountEverything)
* [BMNet+](https://github.com/flyinglynx/Bilinear-Matching-Network)
* [CounTR](https://github.com/Verg-Avesta/CounTR)
* [CACViT](https://github.com/Xu3XiWang/CACViT-AAAI24)
* [LOCA](https://github.com/djukicn/loca)
* [CountGD](https://github.com/Hzzone/PseCo)
* [PseCo](#acknowledgements)

Furthermore, two new models called EasyCounting and MobileCount have been introduced in order the meet some industrial requirements.

## Contents

* [Structure](#structure)
* [Preparation](#preparation)
* [Inference](#inference)
* [Testing Custom Dataset](#testing-custom-dataset)
* [Training](#training)

## Structure

The repo contains multiple folders and sub-folders:

````
$PATH_TO_REPO/
├──── model/
│    ├──── checkpoints/
│    ├──── config/
│    ├──── Results/
│    ├──── demo.py
│    ├──── test.py
│    
├──── plots_and_statistics/
├──── utils/
├──── scripts/
├────.gitignore

````

The model folder EasyCounting must have
* A checkpoints folder containing the model weights
* A Results folder containing all the results from the testing metrics, visualisation, ...
* A config folder containing at least a demoExp.yaml and a testExp.yaml, containing the parameters of the model you which to modify

The model is trainable by running the function main.py. We will explain our to configure
each model later in the repo

## Preparation
### 1. 📦 Download Datasets

In our project, three datasets have been used:

* [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything)
* FSC_indu
* Images_orin

FSC-147 is the same as the official one, but we changed the split file by adding new configurations:
* training_ar_uniform: a sub


### 2. 🚀 Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running the EasyCounting training and inference procedures. To produce the results in the paper, we used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

```
conda create -n easycounting python
conda activate easycounting
cd EasyCounting
pip install -r requirements.txt
```

### 3. Download Pre-Trained Weights

* Make the ```checkpoints``` directory inside the ```model``` repository.

  ```
  mkdir checkpoints
  ```

* Install the pretrained-weights.

  The model weights used in the paper can be downloaded from [Google Drive link](https://drive.google.com/drive/folders/1QjDtryjR_df4ynEq0TA_yxT-f8g6lSgX?usp=drive_link). 

## Inference

To reproduce the results in the paper, run the following commands after activating the Anaconda environment set up in step 4 of [Preparation](#preparation).
For inference, the only thing to change is the ```testExp.yaml``` configuration file in the ```config/``` folder. The DefaultExp is set as an example, please do not change it.

Make sure to enter a valid absolute path of the dataset, and that the following files and folders are present in the dataset folder:

````
$PATH_TO_DATASET/
├──── images/ 
├──── gt_density_map/
├──── split.json
├──── annotation.json
````
There are 4 different configuration of the models:

* EasyCounting-64: DefaultExp with a patch size of 64 and a padding set to true
* MobileCount-64: EasyCounting-64 but setting MODEL__BACKBONE to ```MobileNetV3```
* EasyCounting-32: DefaultExp with NUM_OPE_ITERATIVES_STEPS set to 3, a padding set to true and a rotation set to true
* MobileCount-32: EasyCounting-32 but setting MODEL__BACKBONE to ```MobileNetV3```

Pay attention to also provide the correct checkpoints path, as well as provide the name you want to give to your model.
Make shure to create the ```Results/``` folder specified in the structure of the code, otherwise you will get an error.

For testing just run

```
python test.py 
```

It will run the ```testExp.yaml``` by default. You can provide another config file by using the arg ```-exp``` (without .yaml at the end). Some extra arguments can also be provided like 
* ```-v```: output a visualisation for each image of the dataset, in the Results/name_of_the_model/ folder
* ```-hm```: calculate the OTM F1 score (can take up to 10' per image), provide the results in a hm.csv file
* ```-d```: calculate some metrics and provide informations for each images (mae,mre,avg box size,...), provide the results in a data_statistics.csv file

Example:

```
python test.py -exp EasyCouting32Exp -v -hm -d
```

## Testing Custom Dataset

You can easily test you own dataset, but it has to meet some requirements:
* the dataset folder must have the same structure as mentionned before
* the images must be in .jpeg or .png format, while the density map must be tensor files .npy
* the split.json file and annotation.json msut follow the same configuration as FSC147 and FSCindu

## Training

You can train a model whatever configuration you want, just change the main.exp config file and run

```
python main.py 
```


