import json
import os
import yaml
import subprocess
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import shutil

test_euresys_folder = r"c:\Users\bourezn\Documents\Master_thesis\data\Image_orin\test_euresys"
FSC147_folder = r"c:\Users\bourezn\Documents\Master_thesis\data\FSC147"
Ceramic_folder = r"c:\Users\bourezn\Documents\Master_thesis\data\CeramicCapa"
data_folder = r"c:\Users\bourezn\Documents\Master_thesis\data"

FSC_indu_folder = os.path.join(data_folder, "FSC_indu")
if not os.path.exists(FSC_indu_folder):
    os.makedirs(FSC_indu_folder)

data_folder = FSC147_folder

images_folder           = os.path.join(data_folder, "images_384_VarV2")
gt_folder               = os.path.join(data_folder, "gt_density_map_adaptive_384_384_object_VarV2")
gt2_folder              = os.path.join(data_folder, "gt_density_map_adaptive_384_VarV2")

images_FSC_indu_folder  = os.path.join(FSC_indu_folder, "images")
gt_FSC_indu_folder      = os.path.join(FSC_indu_folder, "gt_density_map_384")
gt2_FSC_indu_folder      = os.path.join(FSC_indu_folder, "gt_density_map")

if not os.path.exists(images_FSC_indu_folder):
    os.makedirs(images_FSC_indu_folder)

if not os.path.exists(gt_FSC_indu_folder):
    os.makedirs(gt_FSC_indu_folder)

if not os.path.exists(gt2_FSC_indu_folder):
    os.makedirs(gt2_FSC_indu_folder)

json_file_annotation = os.path.join(data_folder, "annotation_FSC147_384.json")
new_json_file_annotation = os.path.join(FSC_indu_folder, "annotation.json")
new_json_file_split = os.path.join(FSC_indu_folder, "split.json")
text_file_path = os.path.join(FSC_indu_folder, "test.txt")

if os.path.exists(text_file_path):
    with open(text_file_path, 'r+') as f:
        images_list = f.readlines()
        images_list = [x.strip() for x in images_list]
else:
    images_list = []

with open(json_file_annotation, 'r+') as f:
    old_annotations = json.load(f)

if os.path.exists(new_json_file_annotation):
    with open(new_json_file_annotation, 'r+') as f:
        annotations = json.load(f)
else:
    annotations = {}

if os.path.exists(new_json_file_split):
    with open(new_json_file_split, 'r+') as f:
        split = json.load(f)
else:
    split = {}
    split["test_FSC_indu"] = []

with open(os.path.join(data_folder,"Train_Test_Val_FSC_147.json"),"rb") as f:
    old_split = json.load(f)
    images_ = old_split["test_distinct"]

#images = os.listdir(images_folder)
images = images_

for im in images:
    if im.endswith(".jpg") or im.endswith(".png"):
        # copy image to test_euresys folder
        src = os.path.join(images_folder, im)
        dst = os.path.join(images_FSC_indu_folder, im)
        if os.path.exists(dst):
            img_name, ext = os.path.splitext(im)
            img_name = img_name.split("_")[0]
            # get the number of elements in the folder
            image_number = len(os.listdir(images_FSC_indu_folder))
            new_name = f"{img_name}_{image_number}{ext}"
            dst = os.path.join(images_FSC_indu_folder, new_name)
            annotations[new_name] = old_annotations[im]
            
            old_split = split["test_FSC_indu"]
            old_split.append(new_name)
            split["test_FSC_indu"] = old_split
            images_list.append(new_name)

        else:
            annotations[im] = old_annotations[im]
            split["test_FSC_indu"].append(im)
            images_list.append(im)

        with open(new_json_file_annotation, 'w') as f:
            json.dump(annotations, f, indent=4)

        with open(new_json_file_split, 'w') as f:
            json.dump(split, f, indent=4)

        shutil.copyfile(src, dst)

with open(text_file_path, 'w') as x:
    for image_name in images_list:
        x.write(f"{image_name}\n")

for im in images:

    im = Path(im).stem + ".npy"

    if im.endswith(".npy"):
        # copy image to test_euresys folder
        src = os.path.join(gt_folder, im)
        dst = os.path.join(gt_FSC_indu_folder, im)
        src2 = os.path.join(gt2_folder,im)
        dst2 = os.path.join(gt2_FSC_indu_folder,im)
        if os.path.exists(dst):
            img_name, ext = os.path.splitext(im)
            img_name = img_name.split("_")[0]
            # get the number of elements in the folder
            image_number = len(os.listdir(gt_FSC_indu_folder))
            new_name = f"{img_name}_{image_number}{ext}"

            dst = os.path.join(gt_FSC_indu_folder, new_name)
            dst2 = os.path.join(gt2_FSC_indu_folder, new_name)
        
        shutil.copyfile(src, dst)
        shutil.copyfile(src2, dst2)


                    

