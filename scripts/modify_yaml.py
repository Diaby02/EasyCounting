
from pathlib import Path
from PIL import Image
from os.path import dirname, abspath

import os
import argparse
import yaml
import warnings
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

rootDirectory = Path(dirname(abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='demoExp', help="file name of the config file, ex: DefaultExp, testExp,...")
parser.add_argument('-ps', "--patch_size", type=int, default=32, help="patch size of the model you want to train/test")
parser.add_argument('-m', "--model", type=str, default= "adapted_loca", help="model name from [adapted_loca, loca, cacvit and countgd]", choices=["adapted_loca", "loca", "cacvit", "countgd"])
parser.add_argument('-mn', "--model_name", type=str, help="name you want the model to have") #only for visu
parser.add_argument('-w', "--weights", type=str,help="full path to the weights")
parser.add_argument('-sn',"--saving_name",type=str,help="name you want to give to the weights")
parser.add_argument('-s', "--split", type=str, default="test_FSC_indu", help="split name in split.json")
parser.add_argument('-d', "--device", type=str, default="cuda:0", help="gpu in which you you want to run the process between [cpu,cuda:0,cuda:1]",choices=["cpu","cuda:0","cuda:1"])
parser.add_argument('-e', '--evaluate', action='store_true',help="if 'test' not in your exp, and you still wanna test, you need to pass this argument")
parser.add_argument('-a', "--add_args",  nargs='+')
parser.add_argument('-av', "--add_args_value",  nargs='+')
parser.add_argument('-at', "--add_args_type",  nargs='+')

def main(args):

    yaml_file = os.path.join(rootDirectory.parent,args.model,"config",args.exp+".yaml")
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    if args.evaluate or "test" in args.exp:

        warnings.warn("By default, the test set is FSC_indu. \nIf you need another test, you have to modify the yaml file manually")
    
        if args.model == "adapted_loca":
            config['DATASET']['PATCH_SIZE']      = args.patch_size
            try:
                weights_path = os.path.join(rootDirectory.parent,args.model,args.weights)
                config['MODEL']['MODEL_PATH']        = weights_path
                config['MODEL']["MODEL_NAME"]        = args.model_name
            except:
                ValueError("You need to pass the weights and the model name of the model you would like to evaluate")
        
        else:
            if args.weights:
                weights_path = os.path.join(rootDirectory.parent,args.model,args.weights)
                config['MODEL']['MODEL_PATH']        = weights_path
            
            if args.model_name:
                config['MODEL']["MODEL_NAME"]        = args.model_name

        config['DATASET']['TEST_DATA']       = args.split
        config["TRAINING"]["DEVICE"]         = args.device 

        if args.add_args:
            for i in range(len(args.add_args)):
                if args.add_args_type[i] == "str":
                    config[str(args.add_args[i].split("__")[0])][str(args.add_args[i].split("__")[1])] = str(args.add_args_value[i])
                elif args.add_args_type[i] == "int":
                    config[str(args.add_args[i].split("__")[0])][str(args.add_args[i].split("__")[1])] = int(args.add_args_value[i])
                elif args.add_args_type[i] == "bool":
                    config[str(args.add_args[i].split("__")[0])][str(args.add_args[i].split("__")[1])] = bool(int(args.add_args_value[i]))

    else: 

        warnings.warn("By default, the training set is FSC147. \nIf you need another training, you have to modify the yaml file manually")

        
        config['DATASET']['PATCH_SIZE']      = args.patch_size
        config['DATASET']['TRAINING_DATA']   = args.split
        config["TRAINING"]["DEVICE"]         = args.device

        try:
            config['MODEL']['SAVING_NAME']    = args.saving_name
        except:
            ValueError("You need to pass the weights and the model name of the model you would like to evaluate")

        if args.add_args:
            for i in range(len(args.add_args)):
                if args.add_args_type[i] == "str":
                    config[str(args.add_args[i].split("__")[0])][str(args.add_args[i].split("__")[1])] = str(args.add_args_value[i])
                elif args.add_args_type[i] == "int":
                    config[str(args.add_args[i].split("__")[0])][str(args.add_args[i].split("__")[1])] = int(args.add_args_value[i])
                elif args.add_args_type[i] == "bool":
                    config[str(args.add_args[i].split("__")[0])][str(args.add_args[i].split("__")[1])] = bool(int(args.add_args_value[i]))    

    with open(yaml_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    return

if __name__ == '__main__':
    parser = parser.parse_args()
    main(parser)