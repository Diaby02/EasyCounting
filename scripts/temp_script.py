import os
import numpy as np
import yaml
import subprocess
from pathlib import Path
import json
import torch

orin_folder = "/mnt/grodisk-nvme/Nicolas_student/Image_orin/"
test_data = ["Small_nails_23", "Small_nails_32"]

for images_folder_name in test_data:
    images_folder_p = os.path.join(orin_folder, images_folder_name)
    for folder in os.listdir(images_folder_p):
        if "_" in folder and "all" not in folder:
            data_folder     = os.path.join(images_folder_p, folder)
            json_split_file = os.path.join(data_folder,"split.json")

            with open(json_split_file, 'r') as file:
                annotations = json.load(file)

            new_annotations = {}
            for key in annotations.keys():
                size = folder.split("_")[1]
                new_key = str(images_folder_name)+"_" + str(size)
                new_annotations[new_key] = annotations[key]

            with open(json_split_file, 'w') as file:
                json.dump(new_annotations, file,indent= 4)


            


    

