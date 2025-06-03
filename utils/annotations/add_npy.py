import json
import cv2 # Import the OpenCV library
import matplotlib.pyplot as plt
from annotator import gaussian_filter_density
from pathlib import Path
import os
import numpy as np

#  parameters to change
data_path = r"C:\Users\bourezn\Documents\Master_thesis\data\CeramicCapa"
images_path = os.path.join(data_path,"images")

json_file = os.path.join(data_path,"annotation.json")

with open(json_file,"r") as file :
    annotations = json.load(file)

for key in annotations.keys():
        
    points = annotations[key]["points"]

    density_map = gaussian_filter_density(cv2.imread(os.path.join(images_path,key),-1),points)
    np.save(os.path.join(data_path, "gt_density_map/" + key.split(".")[0]), density_map)