import json
import cv2 # Import the OpenCV library
import matplotlib.pyplot as plt
from pathlib import Path
import os
from annotator import gaussian_filter_density
import numpy as np

#  parameters to change
data_path = r"C:\Users\bourezn\Documents\Master_thesis\data\Image_orin\Small_nails2\images_27\images"
image_of_interest = "Image_9"
multiply_points = 2 #int
axe = "y"           #x or y 
json_file = os.path.join(Path(data_path).parent,"annotation_Small_nails2_27_384.json")

with open(json_file,"r") as file :
    annotations = json.load(file)

points = annotations[image_of_interest]["points"]
new_points = []
for point in points:
    for mul in range(multiply_points):
        if axe == "y":
            new_point = [point[0],point[1]+ (384/multiply_points)*mul]
        else:
            new_point = [point[0]+ (384/multiply_points)*mul,point[1]]

        if not(new_point[0] > 384 or new_point[1] > 384):
            new_points.append(new_point)
    
annotations[image_of_interest]["points"] = new_points

with open(json_file, 'w') as f:
    json.dump(annotations, f, indent=4)

density_map = gaussian_filter_density(cv2.imread(os.path.join(data_path,image_of_interest+".png"),-1),new_points)
np.save(os.path.join(Path(data_path).parent, "gt_density_map/" + image_of_interest), density_map)
plt.imshow(density_map, cmap='viridis', interpolation="nearest")
plt.axis('off')
plt.savefig(os.path.join(Path(data_path).parent, "gt_density_map/" + image_of_interest + "_gt.jpg"), bbox_inches='tight', pad_inches=0)