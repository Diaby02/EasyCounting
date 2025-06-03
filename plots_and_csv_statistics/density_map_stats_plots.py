import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import regex as re
from pathlib import Path
import pandas as pd
import duckdb as db
import numpy as np
from matplotlib.lines import Line2D

from os.path import dirname, abspath
rootDirectory = dirname(abspath(__file__))

results_folders = ["adapted_loca/Results/test_indu_gt","loca/Results/test_FSC_indu_loca_512", "countgd/Results/test_FSC_indu","cacvit/Results/test_FSC_indu","bmnet/Results/test_FSC_indu","adapted_loca/Results/test_FSC_indu_EasyCounting_64_64"]
results_folders_best = ["adapted_loca/Results/test_indu_gt","adapted_loca/Results/test_FSC_indu_EasyCounting_32_32","adapted_loca/Results/test_FSC_indu_EasyCounting_64_64", "adapted_loca/Results/test_FSC_indu_MobileCount_32_32","adapted_loca/Results/test_FSC_indu_MobileCount_64_64"]
results_folders = [os.path.join(Path(rootDirectory).parent,file) for file in results_folders]
patch_sizes = [32,48]
markers = [".","1","^","2","*","s"]
markers2 = [".","s","^","s","^"]
#legend = ["24x24","32x32","48x48","64x64","GT"]
legend = ["GT","LOCA", "CountGD", "CACViT","BMNet+","EasyCounting-64"]
legend2 = ["GT", "EasyCounting-32","EasyCounting-64","MobileCount-32","MobileCount-64"]
colors = [0,1,2,3,4,5]
colors2 = ["darkviolet","blue","blue","red","red"]
#legend_elements = [Line2D([0], [0], marker= '.',color= "C" + str(0), label='24x24'),
#                   Line2D([0], [0], marker= '.',color= "C" + str(1), label='32x32'),
#                   Line2D([0], [0], marker='^', color= "C" + str(2), label='48x48'),
#                   Line2D([0], [0], marker='^', color= "C" + str(3), label='64x64'),
#                   Line2D([0], [0], ls='--', color='black', label='kernel 5'),
#                   Line2D([0], [0], ls=':', color='black', label='kernel 7'),
#                   Line2D([0], [0], color= "C" + str(4), label='GT',markerfacecolor='g', markersize=15)]

#------------------#
# First plot : BBDR
#------------------#

plt.figure(figsize=(5,5))
for i in range(len(results_folders)):

    bbdr_df = pd.read_csv(os.path.join(results_folders[i],"bbdr.csv"))

    if "patch_size" in bbdr_df.columns.values:
        mean_values = [np.average(db.sql(f"SELECT {column_name} from bbdr_df").fetchnumpy()[column_name]) for column_name in bbdr_df[bbdr_df.columns.values[3:]]]
    else:
        mean_values = [np.average(db.sql(f"SELECT {column_name} from bbdr_df").fetchnumpy()[column_name]) for column_name in bbdr_df[bbdr_df.columns.values[2:]]]
        
    box_sizes = [4 + 4*i for i in range(7)]
    plt.plot(box_sizes,mean_values,color="C" + str(colors[i]),marker=markers[i])
    #plt.plot(box_sizes,mean_values,color=colors2[i],marker=markers2[i])

plt.xlabel("Box size (in pixel)")
plt.ylabel("BBDR")
#plt.legend(loc="upper left", handles=legend_elements)
plt.legend(loc="upper left", labels=legend)
plt.savefig(os.path.join(rootDirectory,"bbdr_statistics_sota.pdf"))
plt.close()

#---------------------#
# Second plot : BBMAE
#---------------------#

plt.figure(figsize=(5,5))
for i in range(len(results_folders)):

    bbmre_df = pd.read_csv(os.path.join(results_folders[i],"bbmre.csv"))

    if "patch_size" in bbmre_df.columns.values:
        mean_values = [np.average(db.sql(f"SELECT {column_name} from bbmre_df").fetchnumpy()[column_name]) for column_name in bbmre_df[bbmre_df.columns.values[3:]]]
    else:
        mean_values = [np.average(db.sql(f"SELECT {column_name} from bbmre_df").fetchnumpy()[column_name]) for column_name in bbmre_df[bbmre_df.columns.values[2:]]]
    box_sizes = [4 + 4*i for i in range(7)]
    plt.plot(box_sizes,mean_values,color="C" + str(colors[i]),marker=markers[i])
    #plt.plot(box_sizes,mean_values,color=colors2[i],marker=markers2[i])

plt.xlabel("Box size (in pixel)")
plt.ylabel("BBMAPE")
#plt.legend(loc="lower left", handles=legend_elements)
plt.legend(loc="upper right", labels=legend)

plt.savefig(os.path.join(rootDirectory,"bbdmre_statistics_sota.pdf"))
plt.close()

#-------------------------------#
# Last Table: Hungarian matching
#-------------------------------#


def hm():
    for i in range(len(results_folders)):

        results = []
        results.append(legend[i])
        hm_df = pd.read_csv(os.path.join(results_folders[i],"hm.csv"))

        mean_values_otm     = [round(float(np.average(db.sql(f"SELECT {column_name} from hm_df WHERE method == 'otm'").fetchnumpy()[column_name])),3) for column_name in hm_df[hm_df.columns.values[-4:-1]]]
        #mean_values_gmnp    = [np.average(db.sql(f"SELECT {column_name} from bbmae_df WHERE method == 'gmnp' and patch_size = {patch_sizes[i]}  ").fetchnumpy()) for column_name in hm_df[hm_df.columns.values[4:7]]]
        mean_values_gmns    = [round(float(np.average(db.sql(f"SELECT {column_name} from hm_df WHERE method == 'gmms'").fetchnumpy()[column_name])),3) for column_name in hm_df[hm_df.columns.values[-4:-1]]]
        mean_values_hdbscan = [round(float(np.average(db.sql(f"SELECT {column_name} from hm_df WHERE method == 'hdbscan'").fetchnumpy()[column_name])),3) for column_name in hm_df[hm_df.columns.values[-4:-1]]]
        
        results = results + mean_values_otm + mean_values_gmns + mean_values_hdbscan


        if os.path.exists("statistics_adapted_loca.csv"):
            stat_df = pd.read_csv("statistics_adapted_loca.csv")
            df1 = pd.DataFrame([results], columns=["model","otm_p", "otm_r", "otm_f1", "gmns_p", "gmns_r", "gmns_f1", "hdbscan_p", "hdscan_s", "hdbscan_f1"])
            stat_df = pd.concat([stat_df, df1], ignore_index=True)
        else:
            stat_df = pd.DataFrame([results], columns=["model","otm_p", "otm_r", "otm_f1", "gmns_p", "gmns_r", "gmns_f1", "hdbscan_p", "hdscan_s", "hdbscan_f1"])

        stat_df.to_csv("statistics_adapted_loca.csv",index=False)

def otm_only(results_folders,legend):
    for i in range(len(results_folders)):

        hm_df = pd.read_csv(os.path.join(results_folders[i],"hm.csv"))

        mean_values_otm = float(np.average(db.sql(f"SELECT f1_score from hm_df WHERE method == 'otm'").fetchnumpy()["f1_score"]))
        
        print(legend[i]+": "+str(mean_values_otm))


#----------------------------------------------------------------#
# Add the results of the hungarian matching to the comparison csv
#----------------------------------------------------------------#

def add_hm_results():
    column_otm = []
    column_gmns = []
    column_hdbscan = []
    for i in range(len(results_folders[:-2])):

        results = []
        results.append(patch_sizes[i])
        hm_df = pd.read_csv(os.path.join(results_folders[i],"hm.csv"))

        values_otm      = db.sql(f"SELECT f1_score from hm_df WHERE method == 'otm' and patch_size = {patch_sizes[i]}").fetchnumpy()["f1_score"]
        values_gmns     = db.sql(f"SELECT f1_score from hm_df WHERE method == 'gmms' and patch_size = {patch_sizes[i]}").fetchnumpy()["f1_score"]
        values_hdbscan  = db.sql(f"SELECT f1_score from hm_df WHERE method == 'hdbscan' and patch_size = {patch_sizes[i]}").fetchnumpy()["f1_score"]

        # todo: convert the floats
        column_otm = column_otm + [float(v) for v in values_otm[::-1]]
        column_gmns = column_gmns + [float(v) for v in values_gmns[::-1]]
        column_hdbscan = column_hdbscan + [float(v) for v in values_hdbscan[::-1]]

    df1 = pd.read_csv("input_patch_k3.csv")
            
    df1 = df1.assign(f1_otm=pd.Series(column_otm).values)
    df1 = df1.assign(f1_gmns=pd.Series(column_gmns).values)
    df1 = df1.assign(f1_hdbscan=pd.Series(column_hdbscan).values)

    df1.to_csv("input_patch_k32.csv",index=False)

#---------------------------------------------------------------#
# Time analysis
#---------------------------------------------------------------#

def time_analysis():
    for i in range(len(results_folders)):
        results = []
        results.append(legend[i])
        hm_df = pd.read_csv(os.path.join(results_folders[i],"hm.csv"))
        values_otm      = db.sql(f"SELECT computation_time from hm_df WHERE method == 'otm'").fetchnumpy()["computation_time"]
        values_gmns     = db.sql(f"SELECT computation_time from hm_df WHERE method == 'gmms'").fetchnumpy()["computation_time"]
        values_hdbscan  = db.sql(f"SELECT computation_time from hm_df WHERE method == 'hdbscan'").fetchnumpy()["computation_time"]

        results = results + [float(np.average(v)) for v in [values_otm,values_gmns,values_hdbscan]]

        if os.path.exists("computation_time.csv"):
            stat_df = pd.read_csv("computation_time.csv")
            df1 = pd.DataFrame([results], columns=["model","otm", "gmns", "hdbscan"])
            stat_df = pd.concat([stat_df, df1], ignore_index=True)
        else:
            stat_df = pd.DataFrame([results], columns=["model","otm", "gmns", "hdbscan"])

        stat_df.to_csv("computation_time.csv",index=False)


otm_only(results_folders[2:],legend[2:])

