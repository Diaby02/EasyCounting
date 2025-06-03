import json
import os
import yaml
import subprocess
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

Ceramic_folder = r"c:\Users\bourezn\Documents\Master_thesis\data\CeramicCapa"

json_file_path = os.path.join(Ceramic_folder, "annotation.json")

if os.path.exists(json_file_path):
    with open(json_file_path, 'r+') as f:
        annotations = json.load(f)
else:
    raise Exception("Need to run script_annotation_all_dataset.py before !!!")

images_list = annotations.keys()

text_file_path = os.path.join(Ceramic_folder, "test.txt")
with open(text_file_path, 'w') as x:
    for image_name in images_list:
        x.write(f"{image_name}\n")

print(f"test.txt file created in {Ceramic_folder}")


                    

