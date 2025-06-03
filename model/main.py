from Networks.model import *
import argparse
import yaml
import time

import os
from os.path import dirname, abspath
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


rootDirectory    = dirname(abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='train')


######################################################################################
#
# MAIN PROCEDURE 
# launches an experiment whose parameters are described in a yaml file  
# 
# Example of use in the terminal: python main.py -exp AdaptedLocaExp
# with 'AdaptedLocaExp' beeing the name of the yaml file (in the Todo_list folder) with 
# the wanted configuration 
# 
######################################################################################

def main(parser):
    # -----------------
    # 0. INITIALISATION 
    # -----------------
    # Read the yaml configuration file 
    stream = open(os.path.join(rootDirectory,'config/' + parser.exp + '.yaml'), 'r')
    param  = yaml.safe_load(stream)
    # Path to the folder that will contain results of the experiment
    if param["TRAINING"]["BACKBONE_LR"] > 0:
        add = "_frozen"
    else:
        add= "" 

    resultsPath = os.path.join(rootDirectory, "Results", parser.exp+add)
    checkpointsPath = os.path.join(rootDirectory, "checkpoints", parser.exp+add)

    if not os.path.exists(checkpointsPath):
        os.mkdir(checkpointsPath)

    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    # ------------------------
    # 1. NETWORK INSTANTIATION 
    # ------------------------
    
    myNetwork  = Network_Class(param, resultsPath, checkpointsPath)

    # ------------------
    # 2. TRAIN THE MODEL  
    # ------------------
    
    print(colored('Start to train the network', 'red'))
    # compute time of the training
    start_time = time.time()
    train_loss, valid_loss,_,_ = myNetwork.train()
    print(colored('The network is trained', 'red'))
    end_time = time.time()
    #print the time of the training in hours, minutes and seconds
    print("Total time of the training: ", int((end_time - start_time)/3600), 'h ', int((end_time - start_time)%3600/60), 'min ', int((end_time - start_time)%60), 's')
    time_per_epoch = (end_time - start_time)/param["TRAINING"]["EPOCH"]
    print("Average time per epoch: ", int(time_per_epoch/60), 'min ', int(time_per_epoch%60), 's')

    #nice plot of the train_loss and the valid_loss
    plt.figure()
    plt.plot(train_loss, label='Training loss')
    plt.plot(valid_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Variation of the losses during the training')
    plt.savefig(os.path.join(resultsPath, 'loss_'+ str(param['DATASET']['IMAGE_SIZE'])+'_'+str(param['TRAINING']['EPOCH'])+'_2.pdf'))
    plt.legend()


if __name__ == '__main__':
    parser = parser.parse_args()
    main(parser)

