"""
Demo file for Few Shot Counting>>>>>>>
"""

from Networks.model import Network_Class
from utils.utils_ import getInfoModel, find_peaks
from utils.visu import full_vizu, partial_vizu, simple_vizu
from utils.dataLoader import resize_img

from PIL import Image
from os.path import dirname, abspath
from termcolor import colored

import os
import torch
import argparse
import yaml
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

rootDirectory = dirname(abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='demoExp')
parser.add_argument('-out', type=str, default=os.path.join(rootDirectory, "Results/demoExp"))
parser.add_argument('-fv', '--full_visualization', action='store_true')

def main(parser):
    # -----------------
    # 0. INITIALISATION 
    # -----------------
    # Read the yaml configuration file 
    stream = open(rootDirectory + '\\config\\' + parser.exp + '.yaml', 'r')
    param  = yaml.safe_load(stream)

    # Path to the folder that will contain results of the experiment 
    resultsPath = parser.out

    if not os.path.exists(parser.out):
        os.mkdir(parser.out)
    
    #data of the image
    bbox_file =   param["DATASET"]["IMG_BOX"]
    image_size =  param["DATASET"]["IMAGE_SIZE"]
    input_image = param["DATASET"]["IMG_PATH"]
    num_objects = param["MODEL"]["NUM_OBJECTS"]
    use_first =   param['TRAINING']['USE_FIRST']

    # ------------------------
    # 1. NETWORK INSTANTIATION 
    # ------------------------
    
    myNetwork  = Network_Class(param, resultsPath)

    # ------------------
    # 2. LOAD THE MODEL 
    # ------------------

    myNetwork.loadWeights()

    # ------------------------------------------
    # 3. LOAD THE BOUNDING BOXES
    # ------------------------------------------

    image_name = os.path.basename(input_image)
    image_name = os.path.splitext(image_name)[0]
    
    with open(bbox_file, "r") as fin:
        lines = fin.readlines()

    if len(lines) == 0:
        output_result = os.path.join(resultsPath, "count_" + image_name + ".txt")
        fout = open(output_result, "w")
        fout.write(str(0)+"\n")
        fout.write(str(0))
        fout.close()
        return

    rects1 = list()
    for line in lines:
        data = line.split()
        y1 = int(data[0])
        x1 = int(data[1])
        y2 = int(data[2])
        x2 = int(data[3])
        rects1.append([x1, y1, x2, y2])

    # ---------------------
    # 4. INFERENCE
    # ---------------------

    image = Image.open(input_image).convert("RGB")

    # si on utilise les exemples de la première image
    if use_first:
        if not os.path.isfile(os.path.join(resultsPath, "first_image.txt")):
            Exception("Change use first to true in the demoExp file")
        with open(os.path.join(resultsPath, "first_image.txt"),"r") as f:
            image_first = f.readlines()[0]
        _, boxes = resize_img(Image.open(image_first).convert("RGB"), rects1, image_size, num_objects)
        image, _ = resize_img(image, None, image_size, num_objects)

    else:
        image, boxes = resize_img(image, rects1, image_size, num_objects)
        fout = open(os.path.join(resultsPath, "first_image.txt"), "w")
        fout.write(input_image)
        fout.close()

    image = image.to(myNetwork.get_device())
    boxes = boxes.to(myNetwork.get_device())

    getInfoModel(myNetwork.get_model(), image, boxes)

    print(colored('Start Inference', 'red'))
    start = time.time()
    with torch.no_grad():
        if use_first:
            out, _ = myNetwork.get_model()(image, None)
        else:
            out, _ = myNetwork.get_model()(image, boxes)
    end = time.time()

    if (end - start) > 1:
        print("Total time of infering: ", round((end - start),2), 's')
    else:
        print("Total time of infering: ", round((end - start)*1000,2), 'ms')
        
    pred_cnt = round(out.sum().item())
    print('===> The predicted count is: {:6.2f}'.format(pred_cnt))

    #find peaks
    peaks, pred_cnt_peaks,_,_ = find_peaks(out, boxes)

    #write result in a text file
    output_result = os.path.join(resultsPath, "count_" + image_name + ".txt")
    fout = open(output_result, "w")
    fout.write(str(pred_cnt)+"\n")
    fout.write(str(round((end - start)*1000,2)) +"\n")
    fout.write(str(pred_cnt_peaks))
    fout.close()

    #save density map in a tensor file
    torch.save(out, os.path.join(resultsPath, image_name + "_out.pt"))

    # ---------------------
    # 5. VISUALIZATION
    # ---------------------
    
    if not use_first:
        if parser.full_visualization:
            full_vizu(input_image, out.detach().cpu(), boxes.cpu(), torch.tensor(rects1).to('cpu'), resultsPath,"LOCA",pred_cnt=pred_cnt_peaks, peaks=peaks)
        else:
            partial_vizu(input_image, out.detach().cpu(), boxes.cpu(), torch.tensor(rects1).to('cpu'), resultsPath,"LOCA",pred_cnt=pred_cnt_peaks, peaks=peaks)

    else:
        simple_vizu(input_image, out.detach().cpu(), os.path.join(resultsPath, image_name + "_map.png"), title="LOCA",pred_cnt=pred_cnt, peaks=peaks)


if __name__ == '__main__':
    parser = parser.parse_args()
    main(parser)

