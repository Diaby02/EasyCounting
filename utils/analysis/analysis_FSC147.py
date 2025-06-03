import os
import argparse
from pathlib import Path
import pandas as pd
import yaml
import torch
from os.path import dirname, abspath
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import json
from PIL import Image

import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

# env exec: /home/vision/miniconda3/envs/countgd/bin/python
rootDirectory = dirname(abspath(__file__))

IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

####################################################

# HELPERS

####################################################

def compute_min_size_object(boxes):
    """
    TODO
    """
    min_size = 384
    for j in range(0, boxes.shape[0]):
        x1, y1, x2, y2 = int(boxes[j, 0].item()), int(boxes[j, 1].item()), int(boxes[j, 2].item()), int(boxes[j, 3].item())
        min_size_temp = min(x2 - x1, y2 - y1)
        min_size = min_size_temp if min_size_temp < min_size else min_size

    return min_size

def compute_max_size_object(boxes):
    """
    TODO
    """
    max_size = 0
    for j in range(0, boxes.shape[0]):
        x1, y1, x2, y2 = int(boxes[j, 0].item()), int(boxes[j, 1].item()), int(boxes[j, 2].item()), int(boxes[j, 3].item())
        max_size_temp = max(x2 - x1, y2 - y1)
        max_size = max_size_temp if max_size_temp > max_size else max_size

    return max_size

class TestData():
    def __init__(self, annotations, data_split,im_dir,dt_map_dir, test_data):
        self.img = data_split[test_data]
        self.img_dir = im_dir
        self.annotations = annotations
        self.dt_map_dir = dt_map_dir

    def get_nb_images(self):
        return len(self.img)

    def get_image(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']

        dots = np.array(anno['points'])

        image_path = os.path.join(self.img_dir,im_id)
        image = Image.open(image_path).convert("RGB")
        w,h = image.size
        image =  transforms.Compose([transforms.ToTensor(),transforms.Resize((384, 384))])(image)

        rects = list()
        for bbox in bboxes:
            x1 = int(bbox[0][0])
            y1 = int(bbox[0][1])
            x2 = int(bbox[2][0])
            y2 = int(bbox[2][1])
            rects.append([x1, y1, x2, y2])
        rects = torch.tensor(rects)

        bboxes = (rects / torch.tensor([w,h,w,h])) * 384

        # drawback of the methods: only judge the size of the object based on the bounding boxes: maybe investigate another method

        return image, bboxes, "Image_" + str(im_id), image_path
    
#-------------------------------------------------------------#

# Script for saving statistics about the results

#-------------------------------------------------------------#

def save_data_statistics(bboxes,im_path,image_size,resultpath):
    """
    TODO
    """

    widths = []
    heights = []
    current_bboxes = bboxes.detach().cpu().squeeze(0)
    for j in range(0, current_bboxes.shape[0]):
        x1, y1, x2, y2 = int(current_bboxes[j, 0].item()), int(current_bboxes[j, 1].item()), int(current_bboxes[j, 2].item()), int(current_bboxes[j, 3].item())
        widths.append(x2 - x1)
        heights.append(y2 - y1)

    
    c_name = ["image_name", "resized_size","Avg_ref_h", "Avg_ref_z", "mean_box_size", "std_size", "aspect_ratio"]
    if os.path.exists(os.path.join(resultpath,"info_dataset.csv")):
        df = pd.read_csv(os.path.join(resultpath,"info_dataset.csv"))
    else:
        df = pd.DataFrame(columns=c_name)

    std_width = np.std(widths)
    std_height = np.std(heights)

    avg_width = round(np.average(widths))
    avg_height = round(np.average(heights))

    aspect_ratio = round(avg_height/avg_width,2)

    mean_size = round(np.sqrt((avg_width*avg_height)))
    mean_std = round(np.sqrt((std_width + std_height)),2)

    df2 = pd.DataFrame([["Image_" + str(Path(im_path).stem), image_size,avg_width, avg_height, mean_size, mean_std,aspect_ratio]], columns=c_name)
    df  = pd.concat([df2, df], ignore_index=True)

    df.to_csv(os.path.join(resultpath,"info_dataset.csv"), index=False)


def get_dataset_info(data_set_name,image_size):

    # -----------------
    # 0. INITIALISATION 
    # -----------------

    # Path to the folder that will contain results of the experiment 
    resultsPath = os.path.join(rootDirectory,"analysis_datasets_results", data_set_name)

    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    # load data from FSC147
    data_path       = r"C:\Users\bourezn\Documents\Master_thesis\data\FSC147"
    anno_file       = os.path.join(data_path,'annotation_FSC147_384.json')
    data_split_file = os.path.join(data_path,'Train_Test_Val_FSC_147.json')
    im_dir          = os.path.join(data_path,'images_384_VarV2')
    dt_map_file     = os.path.join(data_path,f'gt_density_map_adaptive_384_384_object_VarV2')
    test_data       = data_set_name

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)

    data = TestData(annotations, data_split, im_dir, dt_map_file, test_data)

    # ------------------------------------------
    # 3. CREATE THE BOUDING BOXES IF NOT GIVEN
    # ------------------------------------------

    pbar = tqdm(range(data.get_nb_images()))
    for img_idx in pbar:
        _, bboxes, _, image_path = data.get_image(img_idx) 
        save_data_statistics(bboxes,image_path, image_size,resultsPath)

def analysis(dataset_name):
    resultsPath = os.path.join(rootDirectory,"analysis_datasets_results", dataset_name)
    df = pd.read_csv(os.path.join(resultsPath,"info_dataset.csv"))

    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    # get all the aspect ratio
    aspect_ratios = df["aspect_ratio"]
    aspect_ratios_mod = [-1/v if v < 1.0 else v for v in aspect_ratios]

    nb_image_b32_and_23 = len([v for v in aspect_ratios if v >= (2/3) and v <= (3/2)])
    print("Number of images between 3:2 and 2:3 aspect ratio: ",nb_image_b32_and_23)
    print("Number of images above 3:2 and 2:3 aspect ratio: ",len(aspect_ratios)-nb_image_b32_and_23)
    print("Number of images above 3:2 aspect ratio: ", len([v for v in aspect_ratios if v > (3/2)]))
    print("Number of images below 2:3 aspect ratio: ", len([v for v in aspect_ratios if v < (2/3)]))

    # get all the object_sizes (resize to the appropriate size)
    object_sizes = df["mean_box_size"]

    print("25, 50 and 75th quantile of the object sizes: ", np.quantile(object_sizes, [0.25, 0.5, 0.75]))

    with open(os.path.join(resultsPath,"results.txt"), "w") as x:

        x.write("Number of images between 3:2 and 2:3 aspect ratio: "+ str(nb_image_b32_and_23) + "\n")
        x.write("Number of images above 3:2 and 2:3 aspect ratio: " + str(len(aspect_ratios)-nb_image_b32_and_23) + "\n")
        x.write("Number of images above 3:2 aspect ratio: " + str(len([v for v in aspect_ratios if v > (3/2)])) + "\n")
        x.write("Number of images below 2:3 aspect ratio: "+ str(len([v for v in aspect_ratios if v < (2/3)])) + "\n")
        x.write("25, 50 and 75th quantile of the object sizes: " + str(np.quantile(object_sizes, [0.25, 0.5, 0.75])))

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,4))

    # rework
    ax1.hist(aspect_ratios_mod, color = 'blue', edgecolor = 'black', bins = 100)
    ax1.set_yscale("log")
    ax1.set_xticks([-30,-20,-10,-1.5,1.5,10,20,30], labels=["1:30","1:20","1:10","2:3","3:2","10:1","20:1","30:1"])
    ax1.set_xlabel("Aspect ratio")
    ax1.set_ylabel("Number of images")

    ax2.hist(object_sizes, color = 'blue', edgecolor = 'black', bins = 100)
    ax2.set_ylabel("Number of images")
    ax2.set_xlabel("Object size")

    plt.tight_layout()
    plt.savefig(os.path.join(resultsPath,"ar_os.png"))

def uniform_dataset(dataset_name,modify_test_set=False):

    resultsPath = os.path.join(rootDirectory,"analysis_datasets_results", dataset_name)
    df = pd.read_csv(os.path.join(resultsPath,"info_dataset.csv"))

    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    # get all the aspect ratio
    aspect_ratios = df["aspect_ratio"]
    df["image_number"] = [im.split("_")[1]+".jpg" for im in df["image_name"]]

    nb_image_b32_and_23 = len([v for v in aspect_ratios if v >= (2/3) and v <= (3/2)])
    nb_image_below_23   = len([v for v in aspect_ratios if v < (2/3)])
    nb_image_above_32   = len([v for v in aspect_ratios if v > (3/2)])

    print("Number of images between 3:2 and 2:3 aspect ratio: ",nb_image_b32_and_23)
    print("Number of images above 3:2 and 2:3 aspect ratio: ",len(aspect_ratios)-nb_image_b32_and_23)
    print("Number of images above 3:2 aspect ratio: ", len([v for v in aspect_ratios if v > (3/2)]))
    print("Number of images below 2:3 aspect ratio: ", len([v for v in aspect_ratios if v < (2/3)]))
    
    min_rep = min([nb_image_below_23, nb_image_b32_and_23, nb_image_above_32])

    subset_1 = df[(df["aspect_ratio"] >= (2/3)) & (df["aspect_ratio"] <= (3/2))].sample(n=min_rep)
    subset_2 = df[df["aspect_ratio"] < (2/3)].sample(n=min_rep)
    subset_3 = df[df["aspect_ratio"] > (3/2)].sample(n=min_rep)

    print(len(subset_1),len(subset_2),len(subset_3))

    new_df = pd.concat([subset_1,subset_2,subset_3], ignore_index=True)

    assert(len(new_df) == 3*min_rep)

    # get all the object_sizes (resize to the appropriate size)
    object_sizes = new_df["mean_box_size"]

    print("25, 50 and 75th quantile of the object sizes: ", np.quantile(object_sizes, [0.25, 0.5, 0.75]))

    if modify_test_set:
        new_split = new_df["image_number"]

        data_path       = r"C:\Users\bourezn\Documents\Master_thesis\data\FSC147"
        data_split_file = os.path.join(data_path,'Train_Test_Val_FSC_147.json')
        with open(data_split_file) as f:
            data_split = json.load(f)

        data_split["train_ar_uniform"]     = new_split.to_list()
        data_split["train_higher_32"]      = subset_3["image_number"].to_list()
        data_split["train_bewteen_23_32"]  = subset_1["image_number"].to_list()
        data_split["train_lower_23"]       = subset_2["image_number"].to_list()

        print(type(data_split))

        with open(os.path.join(resultsPath,"new_data_split.json"), "w") as outfile:
            json.dump(data_split,outfile,indent=4)



def finetune_test(modify_test_set=False):

    resultsPath = os.path.join(rootDirectory,"analysis_datasets_results", "test")
    df = pd.read_csv(os.path.join(resultsPath,"info_dataset.csv"))

    # get all the aspect ratio
    aspect_ratios = df["aspect_ratio"]
    df["image_number"] = [im.split("_")[1]+".jpg" for im in df["image_name"]]

    nb_image_b32_and_23 = len([v for v in aspect_ratios if v >= (2/3) and v <= (3/2)])

    print("Number of images above 3:2 and 2:3 aspect ratio: ",len(aspect_ratios)-nb_image_b32_and_23)

    subset_df = df[(df["aspect_ratio"] < (2/3)) | (df["aspect_ratio"] > (3/2))]
    print(len(subset_df))
    # get all the object_sizes (resize to the appropriate size)
    object_sizes = subset_df["mean_box_size"]

    print("25, 50 and 75th quantile of the object sizes: ", np.quantile(object_sizes, [0.25, 0.5, 0.75]))

    if modify_test_set:
        new_split = subset_df["image_number"]
        #data_path       = r"C:\Users\bourezn\Documents\Master_thesis\data\FSC147"
        #data_split_file = os.path.join(data_path,'Train_Test_Val_FSC_147.json')
        data_split_file = r"C:\Users\bourezn\Documents\Master_thesis\utils\analysis\analysis_datasets_results\train\new_data_split.json"
        with open(data_split_file) as f:
            data_split = json.load(f)

        data_split["train_finetune"]     = new_split.to_list() + data_split["train_ar_uniform"]

        with open(os.path.join(resultsPath,"new_data_split2.json"), "w") as outfile:
            json.dump(data_split,outfile,indent=4)





    

#get_dataset_info("val",384)
#analysis("train")
#uniform_dataset("train")
finetune_test(modify_test_set=True)
