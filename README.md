# Class-Agnostic Object Counting Model for manufacturing applications

Author: Nicolas Bourez

## Introduction

This code aims to compare different class-agnostic counting models in order to understand their capabilities
when applied in industrial data. For this purpose, the code from the different models have been included in this repository.

The link to the official code of the different models are listed below:

* [FamNet+](https://github.com/cvlab-stonybrook/LearningToCountEverything)
* [BMNet+](https://github.com/flyinglynx/Bilinear-Matching-Network)
* [CounTR](https://github.com/Verg-Avesta/CounTR)
* [CACViT](https://github.com/Xu3XiWang/CACViT-AAAI24)
* [LOCA](https://github.com/djukicn/loca)
* [CountGD](https://github.com/Hzzone/PseCo)
* [PseCo](#acknowledgements)

## Contents

* [Structure] (#structure)
* [Preparation](#preparation)
* [CountGD Inference & Pre-Trained Weights](#countgd-inference--pre-trained-weights)
* [Testing Your Own Dataset](#testing-your-own-dataset)
* [CountGD Train](#countgd-train)
* [CountBench](#countbench)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

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

  ```
  TODO
  ```

## EasyCounting Inference & Pre-Trained Weights

The model weights used in the paper can be downloaded from [Google Drive link (1.2 GB)](https://drive.google.com/file/d/1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI/view?usp=sharing). To reproduce the results in the paper, run the following commands after activating the Anaconda environment set up in step 4 of [Preparation](#preparation). Make sure to change the directory and file names in [datasets_fsc147_val.json](https://github.com/niki-amini-naieni/CountGD/blob/main/config/datasets_fsc147_val.json) and [datasets_fsc147_test.json](https://github.com/niki-amini-naieni/CountGD/blob/main/config/datasets_fsc147_test.json) to the ones you set up in step 1 of [Preparation](#preparation). Make sure that the model file name refers to the model that you downloaded.

For the validation set (takes ~ 26 minutes on 1 RTX 3090 GPU):

```
python -u main_inference.py --output_dir ./countgd_val -c config/cfg_fsc147_val.py --eval --datasets config/datasets_fsc147_val.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased --crop --sam_tt_norm --remove_bad_exemplar
```

For the validation set with no Segment Anything Model (SAM) test-time normalization and, hence, slightly reduced counting accuracy (takes ~ 6 minutes on 1 RTX 3090 GPU):

```
python -u main_inference.py --output_dir ./countgd_val -c config/cfg_fsc147_val.py --eval --datasets config/datasets_fsc147_val.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased --crop --remove_bad_exemplar
```

For the test set (takes ~ 26 minutes on 1 RTX 3090 GPU):

```
python -u main_inference.py --output_dir ./countgd_test -c config/cfg_fsc147_test.py --eval --datasets config/datasets_fsc147_test.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased --crop --sam_tt_norm --remove_bad_exemplar
```

For the test set with no Segment Anything Model (SAM) test-time normalization and, hence, slightly reduced counting accuracy (takes ~ 6 minutes on 1 RTX 3090 GPU):

```
python -u main_inference.py --output_dir ./countgd_test -c config/cfg_fsc147_test.py --eval --datasets config/datasets_fsc147_test.json --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth --options text_encoder_type=checkpoints/bert-base-uncased --crop --remove_bad_exemplar
```

* Note: Inference can be further sped up by increasing the batch size for evaluation

## Testing Your Own Dataset

You can run CountGD on all the images in a zip folder uploaded to Google Drive using the Colab notebook [here](https://github.com/niki-amini-naieni/CountGD/blob/main/google-drive-batch-process-countgd.ipynb) in the repository or [here](https://huggingface.co/spaces/nikigoli/countgd/blob/main/notebooks/demo.ipynb) online. This code supports a single text description for the whole dataset but can be easily modified to handle different text descriptions for different images and to support exemplar inputs.

## CountGD Train

See [here](https://github.com/niki-amini-naieni/CountGD/blob/main/training.md) for the code and [here](https://github.com/niki-amini-naieni/CountGD/issues/32) about a relevant issue

## CountBench

See [here](https://github.com/niki-amini-naieni/CountGD/issues/6)

## Citation
If you use our research in your project, please cite our paper.

```
@InProceedings{AminiNaieni24,
  author = "Amini-Naieni, N. and Han, T. and Zisserman, A.",
  title = "CountGD: Multi-Modal Open-World Counting",
  booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
  year = "2024",
}
```

### Acknowledgements

This repository is based on the [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino/tree/main) and uses code from the [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO). If you have any questions about our code implementation, please contact us at [niki.amini-naieni@eng.ox.ac.uk](mailto:niki.amini-naieni@eng.ox.ac.uk).

# Bilinear Matching Network

This repository is the official implementation of our CVPR 2022 Paper "Represent, Compare, and Learn: A Similarity-Aware Framework for Class-Agnostic Counting". [Link](https://arxiv.org/abs/2203.08354)

In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022
Min Shi, Hao Lu, Chen Feng, Chengxin Liu, Zhiguo Cao<sup>*</sup>

Key Laboratory of Image Processing and Intelligent Control, Ministry of Education
School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, China
<sup>*</sup>  corresponding author.

## Updates
- We are currently organizing a more detailed readme file, with more instructions and discussions on how to build a strong baseline for class-agnostic counting. You can first explore our codes. Feel free to post your questions!
- 23 Apr 2022: Training and inference code is released.

## Installation
Our code has been tested on Python 3.8.5 and PyTorch 1.8.1+cu111. Please follow the official instructions to setup your environment. See other required packages in `requirements.txt`.

## Data Preparation
We train and evaluate our methods on FSC-147 dataset. Please follow the [FSC-147 official repository](https://github.com/cvlab-stonybrook/LearningToCountEverything) to download and unzip the dataset. Then, please place the data lists  ``data_list/train.txt``, ``data_list/val.txt`` and ``data_list/test.txt`` in the dataset directory. Note that, you should also download data annotation file ``annotation_FSC147_384.json`` and ``ImageClasses_FSC147.txt`` file from [Link](https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master/data) and place them in the dataset folder. Final the path structure used in our code will be like :


