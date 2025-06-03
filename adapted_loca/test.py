from Networks.model import Network_Class
import os
import argparse
import yaml
from os.path import dirname, abspath
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

# env exec: /mnt/grodisk/Nicolas_student/myvenv/bin/python3.12
rootDirectory = dirname(abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='testExp')
parser.add_argument('-out', type=str, default=os.path.join(rootDirectory, "Results/"))
parser.add_argument('-v', '--visu', action="store_true")
parser.add_argument('-p', '--peaks', action="store_true")
parser.add_argument('-sd', '--save_data', action="store_true")
parser.add_argument('-hm', '--hungarian_matching', action="store_true")

def main(parser):
    # -----------------
    # 0. INITIALISATION 
    # -----------------
    # Read the yaml configuration file 
    stream = open(rootDirectory + '/config/' + parser.exp + '.yaml', 'r')
    param  = yaml.safe_load(stream)
    # Path to the folder that will contain results of the experiment 
    
    resultsPath = parser.out + param["DATASET"]["TEST_DATA"] + "_" + param["MODEL"]["MODEL_NAME"] + "_" + str(param["DATASET"]["PATCH_SIZE"])

    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    # ------------------------
    # 1. NETWORK INSTANTIATION 
    # ------------------------
    
    #print("==> Preparing the network...")
    myNetwork  = Network_Class(param, resultsPath)

    # ------------------
    # 2. LOAD THE MODEL 
    # ------------------

    myNetwork.loadWeights()
    
    # --------------------
    # 3. EVALUATE
    #---------------------

    myNetwork.evaluate(visu=parser.visu, peaks_bool=parser.peaks, save_data=parser.save_data, hm=parser.hungarian_matching)



if __name__ == '__main__':
    #print("==> Start the loca demo...")
    parser = parser.parse_args()
    main(parser)