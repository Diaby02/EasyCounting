import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import regex as re
from pathlib import Path

dir_path = r"c:\Users\bourezn\Documents\Master_thesis\data\CeramicCapa"
dir_path_inference = os.path.join(dir_path, "results_inference")
models_name = [str(Path(f).stem) for f in os.listdir(dir_path_inference)]
models_true_names = ["BMNet+", "CACViT", "CountGD", "CounTR", "LOCA", "FamNet+", "PseCo"]

def get_mae_and_inftime(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    mae = 0
    time = 0
    for line in lines:
        if "MAE" in line:
            regex = r"(\d+.\d+)"
            matches = re.findall(regex, line, re.MULTILINE)
            mae = matches[0]

        if "time" in line:
            regex = r"(\d+.\d+\ss|\d+.\d+\sms)"
            matches = re.findall(regex, line, re.MULTILINE)
            time = matches[0]
            

    return round(float(mae), 2), time

def convert_all_time(results_time):

    new_time = []
    for time in results_time:
        regex = r"(\d+.\d+)"
        matches = re.findall(regex, time, re.MULTILINE)
        time_int= round(float(matches[0]),3)

        
        if "ms" in time:
            time_int = round(time_int/1000, 2)

        new_time.append(time_int)

    return new_time


results_mae = [get_mae_and_inftime(os.path.join(dir_path_inference,f, str(Path(f).stem) + ".txt"))[0] for f in os.listdir(dir_path_inference)]
results_time = convert_all_time([get_mae_and_inftime(os.path.join(dir_path_inference,f, str(Path(f).stem) + ".txt"))[1] for f in os.listdir(dir_path_inference)])
results_mae_dict = {}
results_time_dict = {}
for i in range(len(results_mae)):
    results_mae_dict[models_true_names[i]] = results_mae[i]
    results_time_dict[models_true_names[i]] = results_time[i]

#######################################################################

# PLOTTING TIME

#######################################################################

def plotiplot(results_dict, title, dataset_name, unit=None):
    x_labels = [k for k in results_dict.keys()]
    y_val = [v for v in results_dict.values()]

    colors_dict = {"FamNet+": "green", "BMNet+" : "green", "CountGD":"red", "CounTR":"yellow", "LOCA" : "green", "CACViT":"yellow", "PseCo":"red"}
    bar_labels_dict = {"FamNet+":"ResNet50", "BMNet+" : "ResNet50",  "CountGD":"VLMs","CounTR": "ViT", "LOCA" : "ResNet50", "CACViT": "ViT", "PseCo":"VLMs"}

    sorted_colors = []
    sorted_labels = []
    for k,v in results_dict.items():
        sorted_colors.append(colors_dict[k])

        if bar_labels_dict[k] in sorted_labels:
            sorted_labels.append("_" + bar_labels_dict[k])
        else:
            sorted_labels.append(bar_labels_dict[k])


    bar_colors = sorted_colors
    bar_labels = sorted_labels

    fig, ax = plt.subplots(figsize=(8.8, 4), layout="constrained")

    ax.plot(range(0,len(x_labels)), y_val, color="blue", linestyle= "-", linewidth=1.5, zorder=1)

    for i, (mae, color) in enumerate(zip(y_val,sorted_colors)):
        ax.scatter(i, mae, color=color, edgecolors="black", s=100, zorder=2)

    for i, txt in enumerate(y_val):
        if title == "Inference time":
            val = (max(y_val)/5)/(i*10+1)
            ax.annotate(str(txt), (i, txt+val))
        else:
            ax.annotate(str(txt), (i, txt+(max(y_val)/25)))

    #ax.bar(x_labels, y_val, label=bar_labels, color=bar_colors)
    ax.margins(0.06,0.1)
    if title == "Inference time":
        ax.set_yscale("log")
    ax.set_xticks(range(0,len(x_labels)))
    ax.set_xticklabels(x_labels)
    if unit:
        ax.set_ylabel(title+" (" + unit + ")")
    else:
        ax.set_ylabel(title)
    ax.set_xlabel("Models")
    #ax.set_title(title + " of the selected FSC-models on the " + dataset_name + " dataset")

    #custom legend

    red_patch = mpatches.Patch(color="red", label="VLMs")
    green_patch = mpatches.Patch(color="green", label="CNN")
    yellow_patch = mpatches.Patch(color="yellow", label="ViT")

    ax.legend(handles=[green_patch,yellow_patch, red_patch],loc="upper right")

    #sorted_colors = ["red" for _ in range(3)]
    #for i, (mae, color) in enumerate(zip(y_val[0:3],sorted_colors)):
    #    ax.scatter(i, mae, color=color, edgecolors="red", s=300, zorder=50, linewidths=5, marker='x')

    plt.savefig("plots/"+ dataset_name +"_ "+ title + ".pdf")
    plt.show()

# MAE
# Countgd: 7.024
measure_mae_FSC_indu = {"BMNet+": 17.35, "CACViT": 8.46, "CountGD": 5.35, "CounTR": 12.22, "LOCA": 6.74, "FamNet+": 18.62, "PseCo": 15.37}
measure_time_FSC_indu = {"BMNet+": 0.028, "CACViT": 0.071, "CountGD": 0.23, "CounTR": 0.057, "LOCA": 0.040, "FamNet+": 0.032, "PseCo": 4.38}

#sorted_dict = dict(sorted(results_mae_dict.items(), key=lambda item: item[1], reverse=True))
sorted_dict = dict(sorted(measure_mae_FSC_indu.items(), key=lambda item: item[1], reverse=True))
plotiplot(sorted_dict, "Average MAE", "FSCindu")

#TIME
#sorted_dict = dict(sorted(results_time_dict.items(), key=lambda item: item[1], reverse=True))
sorted_dict = dict(sorted(measure_time_FSC_indu.items(), key=lambda item: item[1], reverse=True))
plotiplot(sorted_dict, "Inference time", "FSCindu", unit="s")
