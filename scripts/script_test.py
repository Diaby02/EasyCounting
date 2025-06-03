import os
import yaml
import subprocess
from pathlib import Path
from os.path import dirname, abspath
rootDirectory = dirname(abspath(__file__))
import argparse

main_folder = "/mnt/grodisk-nvme/Nicolas_student"
MODEL_DIRECTORY = "/mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca"
ENV_PATH = "/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python3"
TEST_FOLDER = [os.path.join(main_folder,"FSC147")]
#TEST_FOLDER = [os.path.join(main_folder,"FSC147"), os.path.join(main_folder,"FSC147"), os.path.join(main_folder,"FSC_indu")]
TEST_NAME = ["test"]
#TEST_NAME = ["val", "test", "test_FSC_indu"]

list_of_checkpoints = os.listdir(os.path.join(MODEL_DIRECTORY,"checkpoints/AdaptedLocaExp"))
list_of_checpoints_of_interest = [checkpoint for checkpoint in list_of_checkpoints if "_p_ope" in checkpoint]

rootDirectory = dirname(abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='testExp')

def process_test(exp):
    """Iterate through images folders and run test.py for each one."""

    test_file_path = os.path.join(MODEL_DIRECTORY, "test.py")
    env_python = os.path.join(ENV_PATH)
    yaml_path = os.path.join(MODEL_DIRECTORY, "config/"+exp+".yaml")

    for checkpoint_name in list_of_checpoints_of_interest:

        print("Starting the testing of ", checkpoint_name)

        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        config["TRAINING"]["DEVICE"]         = "cuda:0"
        config["MODEL"]["MODEL_PATH"]        = os.path.join(MODEL_DIRECTORY,"checkpoints/AdaptedLocaExp/"+ checkpoint_name)
        config["MODEL"]["MODEL_NAME"]        = checkpoint_name.replace(".pt","")

        if "_p_" in checkpoint_name:
            config["MODEL"]["PADDING"]           = True

        if "32" in checkpoint_name:
            config["DATASET"]["PATCH_SIZE"] = 32
        
        if "48" in checkpoint_name:
            config["DATASET"]["PATCH_SIZE"] = 48

        if "64" in checkpoint_name:
            config["DATASET"]["PATCH_SIZE"] = 64

        if "ope" in checkpoint_name:
            if "00" in checkpoint_name:
                config["MODEL"]["SCALE_ONLY"] = False
                config["MODEL"]["SCALE_AS_KEY"] = False
            if "01" in checkpoint_name:
                config["MODEL"]["SCALE_ONLY"] = False
                config["MODEL"]["SCALE_AS_KEY"] = True
            if "10" in checkpoint_name:
                config["MODEL"]["SCALE_ONLY"] = True
                config["MODEL"]["SCALE_AS_KEY"] = False
            if "11" in checkpoint_name:
                config["MODEL"]["SCALE_ONLY"] = True
                config["MODEL"]["SCALE_AS_KEY"] = True

        with open(yaml_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        for i, test_dir in enumerate(TEST_FOLDER):

            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)

            config['DATASET']['DATA_PATH']       = str(test_dir)
            config['DATASET']['IMAGE_FOLDER']    = "images"

            if "FSC147" in test_dir:
                config['DATASET']['GT_FOLDER']       = "gt_density_map_adaptive_384_384_object_VarV2"
            else:
                config['DATASET']['GT_FOLDER']       = "gt_density_map"

            config['DATASET']['SPLIT_FILE']      = "split.json"
            config['DATASET']['ANNOTATION_FILE'] = "annotation.json"

            
            config['DATASET']['TEST_DATA']       = TEST_NAME[i]

            with open(yaml_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            
            subprocess.run([env_python, test_file_path, "-exp", exp])

    return

if __name__ == "__main__":
    parser = parser.parse_args()
    process_test(parser.exp)

