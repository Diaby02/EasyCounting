import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import regex as re
from pathlib import Path

dir_path = "/mnt/grodisk/Nicolas_student/deepcounting/adapted_loca/Results/DefaultExp/"
result_file = os.path.join(dir_path, "results_2.txt")
stats_file  = os.path.join(dir_path, "stats_2.txt")

with open(result_file, "r") as file:
    lines = file.readlines()

train_loss = []
val_loss  = []
nb_epochs = range(len(lines))

for line in lines[2:]:
    line_split = line.split("\t")
    train_loss.append(float(line_split[0]))
    val_loss.append(float(line_split[1]))

with open(stats_file, "r") as file:
    lines = file.readlines()

train_MAE = []
val_MAE  = []
nb_epochs = range(len(lines))

for line in lines:
    line_split = line.split(",")
    train_MAE.append(float(line_split[0]))
    val_MAE.append(float(line_split[1]))

#######################################################################

# PLOTTING TIME

#######################################################################

plt.figure(figsize=(9,3))
plt.plot(nb_epochs, train_MAE, c='blue', label="Train MAE")
plt.plot(nb_epochs, val_MAE, c='orange', label="Val MAE")


plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Evolution of the MAE through epochs")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("FSC147_adapted_loca_50_MAE_"+model+".pdf")
plt.show()

plt.figure(figsize=(9,3))
plt.plot(nb_epochs, train_loss, c='blue', label="Train loss")
plt.plot(nb_epochs, val_loss, c='orange', label="Val loss")

plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.title("Evolution of the loss through epochs")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("FSC147_adapted_loca_50_loss_"+model+".pdf")
plt.show()

