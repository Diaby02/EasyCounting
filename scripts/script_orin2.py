import os
import yaml
import subprocess
from pathlib import Path
from os.path import dirname, abspath
rootDirectory = dirname(abspath(__file__))
import argparse

orin_folder = "/mnt/grodisk-nvme/Nicolas_student/Image_orin/"
MODELS_DIRECTORY = "/mnt/grodisk-nvme/Nicolas_student/deepcounting"
ENV_PATH = "/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python3"
ENV_CACVIT = "/usr/bin/python"
ENV_COUNTGD = "/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python3"

rootDirectory = dirname(abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='testExp2')
parser.add_argument('-p', "--pattern", type=str, default="_")
parser.add_argument('-m', "--model", type=str, default= "adapted_loca")
parser.add_argument('-t', "--test_data",  nargs='+', default= ["Nails_23","Nails_32", "Nuts", "Pins2", "Small_nails_23", "Small_nails_32"])
parser.add_argument('-hm', "--hungarian_matching", action="store_true")
parser.add_argument('-v', "--visu", action="store_true")

def process_images(model,exp,test_data,pattern='_',visu=False,hm=False):
    """Iterate through images folders and run test.py for each one."""

    test_file_path = os.path.join(MODELS_DIRECTORY, model, "test.py")
    if model == "cacvit":
        env_python = os.path.join(ENV_CACVIT)
    elif model == "countgd":
        env_python = os.path.join(ENV_COUNTGD)
    else:
        env_python = os.path.join(ENV_PATH)
    yaml_path = os.path.join(MODELS_DIRECTORY, model + "/config/"+exp+".yaml")

    for images_folder_name in test_data:
        images_folder_p = os.path.join(orin_folder, images_folder_name)
        for folder in os.listdir(images_folder_p):
            if pattern in folder and "all" not in folder:
                data_folder     = os.path.join(images_folder_p, folder)

                with open(yaml_path, 'r') as file:
                    config = yaml.safe_load(file)

                config['DATASET']['DATA_PATH']       = str(data_folder)
                config['DATASET']['GT_FOLDER']       = "gt_density_map"
                config['DATASET']['IMAGE_FOLDER']    = "images"
                config['DATASET']['SPLIT_FILE']      = "split.json"
                config['DATASET']['ANNOTATION_FILE'] = "annotation.json"
                config['DATASET']['TEST_DATA']       = images_folder_name + "_" + str(folder.split("_")[1])
                config["TRAINING"]["DEVICE"]         = "cuda:0"

                with open(yaml_path, 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        
                if hm :
                    if visu:
                        subprocess.run([env_python, test_file_path, "-exp", exp,"-hm","-v", "-sd"])
                    else: 
                        subprocess.run([env_python, test_file_path, "-exp", exp,"-hm", "-sd"])
                else:
                    if visu:
                        subprocess.run([env_python, test_file_path, "-exp", exp,"-v", "-sd"])
                    else: 
                        subprocess.run([env_python, test_file_path, "-exp", exp, "-sd"])

    return

if __name__ == "__main__":
    parser = parser.parse_args()
    test_data = [str(t) for t in parser.test_data]
    process_images(parser.model,parser.exp,test_data,parser.pattern,parser.visu,parser.hungarian_matching)

