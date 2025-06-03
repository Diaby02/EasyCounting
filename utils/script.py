import os
import yaml
import subprocess
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

MODELS_DIRECTORY = r"C:\Users\bourezn\Documents\Master_thesis\deepcounting"
CUDA_ENV_PATH = r"C:/Users/bourezn/AppData/Local/miniconda3/envs"
BOX_PATH_FIRST = ""

def update_yaml(yaml_path, image_path,box_path=None, first=False,use_first=True):
    """Update the IMG_PATH in the YAML file."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    if first == True:
        use_first=False
    else:
        use_first=use_first
    
    config['DATASET']['IMG_PATH'] = str(image_path)
    config['TRAINING']['FIRST'] = first
    config['TRAINING']['USE_FIRST'] = use_first
    
    if box_path:
        config['DATASET']['IMG_BOX'] = str(box_path)
    else:   
        config['DATASET']['IMG_BOX'] = None
    
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def process_images(image_dir, model, env, only_first, box_dir=None, im_type="png"):
    """Iterate through images and run demo.py for each one."""

    # create all the usefull paths
    image_dir = Path(image_dir)
    parent_dir_images = image_dir.parent

    script_path = os.path.join(MODELS_DIRECTORY, model, "demo.py")
    parent_dir_script = Path(script_path).parent

    env_python = os.path.join(CUDA_ENV_PATH, env + "/python.exe")
    yaml_path = os.path.join(MODELS_DIRECTORY, model + r"\config\demoExp.yaml")

    if only_first:
        print("HERE")
        result_dir = os.path.join(parent_dir_images, "results_inference_one_example", model)
    else:
        result_dir = os.path.join(parent_dir_images, "results_inference", model)
    # create the directory if it does not exist yet
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    #check if the directory exists
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist.")
        return

    
    list_mae = {}
    list_mae_peaks = {}
    list_seconds = []

    skip = False
    pattern = "*." + im_type

    for i,image_path in enumerate(image_dir.glob(pattern)):  # Adjust extension if needed
        print(f"Processing {image_path}...")
        image_name = image_path.stem
        if "rods" in image_name:
            continue
        if box_dir:
            box_path = os.path.join(box_dir, image_name + "_box.txt")
        else:
            box_path = None
        
        if only_first:
            if i == 0:
                BOX_PATH_FIRST = box_path
                update_yaml(yaml_path, image_path,box_path=box_path,first=True)
            else:
                update_yaml(yaml_path, image_path,box_path=BOX_PATH_FIRST)
        else:
            BOX_PATH_FIRST = box_path
            update_yaml(yaml_path, image_path,box_path=box_path,first=False, use_first=False)


        # Activate the environment and execute demo.py
        #if not skip and not os.path.exists(os.path.join(result_dir, "count_" + image_name + ".txt")):
        if not skip:
            subprocess.run([env_python, script_path, "-exp", "demoExp", "-out", result_dir])

        # get the predicted count
        with open(os.path.join(result_dir, "count_" + image_name + ".txt"), "r") as fin:
            lines = fin.readlines()
        
        lines[0] = lines[0].replace("\n", "")
        predicted_count = int(lines[0])
        predicted_time = float(lines[1])
        predicted_peaks = predicted_count

        if len(lines) > 2:
            lines[2] = lines[2].replace("\n", "")
            predicted_peaks = int(lines[2])

        if predicted_peaks > 2*predicted_count:
            predicted_peaks = predicted_count

        #get the true count
        coordinates_path = os.path.join(parent_dir_images, "points_coordinates/" + image_name + "_annotated.csv")
        coord_df = pd.read_csv(coordinates_path)
        nb_points = len(coord_df.index)

        mae = np.abs(predicted_count - nb_points)
        mae_peaks = np.abs(predicted_peaks - nb_points)
        
        print("MAE on image "+ image_name +": "+ str(mae))
        print("MAE after NMS on image "+ image_name +": "+ str(mae_peaks))
        list_mae[image_name] = mae
        list_mae_peaks[image_name] = mae_peaks
        list_seconds.append(predicted_time)

    avg_mae = np.average([v for v in list_mae.values()])
    avg_mae_peaks = np.average([v for v in list_mae_peaks.values()])
    avg_time = np.average(list_seconds)

    if avg_time > 1000:
        avg_time = str(round(avg_time/1000,2)) + " s"
    else:
        avg_time = str(avg_time) + " ms"

    out_rslt_file = os.path.join(result_dir, model + ".txt")
    fout = open(out_rslt_file, "w")

    for k, v in list_mae.items():
        fout.write("{} :{}\n".format(k,v))

    fout.write("\n")
    for k, v in list_mae_peaks.items():
        fout.write("{} :{}\n".format(k,v))

    fout.write("Average MAE: " + str(avg_mae) +"\n")
    fout.write("Average MAE after NMS: " + str(avg_mae_peaks) +"\n")
    fout.write("Average inference time: " + avg_time)

    fout.close()

    return        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', default=r"C:\Users\bourezn\Documents\Master_thesis\data\CeramicCapa\images", help="Path to the image folder")
    parser.add_argument('-m', '--model', default="loca", help="name of the model", choices=["loca", "ltce","cacvit","countr","DAVE","GeCo","bmnet","pseco"])
    parser.add_argument('-e', '--env', default="loca", help="name to the environment")
    parser.add_argument('-f', '--first', action='store_true', help="take the boxes of the first image")
    parser.add_argument('-it', '--image_type', type=str, default="png")
    args = parser.parse_args()

    #C:\Users\bourezn\Documents\Master_thesis\data\Halcon_test\images

    """ example if you have the boxes
    
    conda activate loca
    python script.py
    
    """

    image_dir = args.image_dir
    model = args.model
    env = args.env
    im_type = args.image_type
    only_first= args.first
    box_dir = os.path.join(Path(image_dir).parent, "bboxes")
    if not os.path.isdir(box_dir):
        print("You don't have any boxes !")
        box_dir = None
    
    print("Processing images...")
    process_images(image_dir, model, env, only_first, box_dir=box_dir, im_type=im_type)