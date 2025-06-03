import json
import os
import yaml
import subprocess
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import shutil

orin_folder = r"c:\Users\bourezn\Documents\Master_thesis\data\Image_orin"
test_data = ["Nails_23", "Nails_32","Pins2","Nuts","Small_nails_23", "Small_nails_32"]
test_euresys_folder = os.path.join(orin_folder, "test_euresys")
if not os.path.exists(test_euresys_folder):
    os.makedirs(test_euresys_folder)

for images_folder_name in test_data:
    images_folder_p = os.path.join(orin_folder, images_folder_name)
    for folder in os.listdir(images_folder_p):
        if "_" in folder and "all" not in folder:
            data_folder     = os.path.join(images_folder_p, folder)

            images_folder           = os.path.join(data_folder, "images")
            gt_folder               = os.path.join(data_folder, "gt_density_map")

            images_euresys_folder = os.path.join(test_euresys_folder, "images")
            gt_euresys_folder = os.path.join(test_euresys_folder, "gt_density_map")

            if not os.path.exists(images_euresys_folder):
                os.makedirs(images_euresys_folder)

            if not os.path.exists(gt_euresys_folder):
                os.makedirs(gt_euresys_folder)

            json_file_annotation = os.path.join(data_folder, "annotation.json")
            new_json_file_annotation = os.path.join(test_euresys_folder, "annotation.json")

            new_json_file_split = os.path.join(test_euresys_folder, "split.json")

            text_file_path = os.path.join(test_euresys_folder, "test.txt")

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
                split["test_euresys"] = []


            for im in os.listdir(images_folder):
                if im.endswith(".jpg") or im.endswith(".png"):
                    # copy image to test_euresys folder
                    src = os.path.join(images_folder, im)
                    dst = os.path.join(images_euresys_folder, im)
                    if os.path.exists(dst):
                        img_name, ext = os.path.splitext(im)
                        img_name = img_name.split("_")[0]
                        # get the number of elements in the folder
                        image_number = len(os.listdir(images_euresys_folder))
                        new_name = f"{img_name}_{image_number}{ext}"
                        dst = os.path.join(images_euresys_folder, new_name)
                        annotations[new_name] = old_annotations[im]
                        
                        old_split = split["test_euresys"]
                        old_split.append(new_name)
                        split["test_euresys"] = old_split
                        images_list.append(new_name)

                    else:
                        annotations[im] = old_annotations[im]
                        split["test_euresys"].append(im)
                        images_list.append(im)

                    with open(new_json_file_annotation, 'w') as f:
                        json.dump(annotations, f, indent=4)

                    with open(new_json_file_split, 'w') as f:
                        json.dump(split, f, indent=4)

                    shutil.copyfile(src, dst)

            with open(text_file_path, 'w') as x:
                for image_name in images_list:
                    x.write(f"{image_name}\n")

            for im in os.listdir(gt_folder):

                if im.endswith(".npy"):
                    # copy image to test_euresys folder
                    src = os.path.join(gt_folder, im)
                    dst = os.path.join(gt_euresys_folder, im)
                    if os.path.exists(dst):
                        img_name, ext = os.path.splitext(im)
                        img_name = img_name.split("_")[0]
                        # get the number of elements in the folder
                        image_number = len(os.listdir(gt_euresys_folder))
                        new_name = f"{img_name}_{image_number}{ext}"

                        dst = os.path.join(gt_euresys_folder, new_name)
                    
                    shutil.copyfile(src, dst)


                    

