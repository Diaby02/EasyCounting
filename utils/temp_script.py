import json
import os
from pathlib import Path
from PIL import Image

path_to_data = r"c:\Users\bourezn\Documents\Master_thesis\data\FSC147"
path_to_images = os.path.join(path_to_data,"images")
json_file = os.path.join(path_to_data, "annotation_FSC147_384.json")

with open(os.path.join(path_to_data, "test_distinct.txt"), 'r') as f:
    image_names = f.readlines()

with open(json_file, 'r+') as f:
    annotations = json.load(f)

max_count = 0
min_count = 1000000
for image_name in image_names:
    image_name = image_name.strip()
    if len(annotations[image_name]["points"]) > max_count:
        max_count = len(annotations[image_name]["points"])
    if len(annotations[image_name]["points"]) < min_count:
        min_count = len(annotations[image_name]["points"])

print("Max count: ", max_count)
print("Min count: ", min_count)

