import os
from matplotlib import pyplot as plt
import yaml
import subprocess
from pathlib import Path
import pandas as pd
import duckdb as db
import numpy as np
import argparse

from os.path import dirname, abspath
rootDirectory = dirname(abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--file", type=str, default="hm.csv")
parser.add_argument('-r', "--result_folder", type=str, default="adapted_loca/Results")
parser.add_argument('-s', "--save_name", type=str)
parser.add_argument('-end', "--end_name", type=str)
parser.add_argument('-ht', "--heat_map", action="store_true")

deepcounting_folder = Path(rootDirectory).parent
result_folder = ""
test_data = ["Nails_23","Nails_32","Pins2","Nuts","Small_nails_23", "Small_nails_32"]

def create_big_csv(file, save_name, end_name):

    file_columns = pd.read_csv(os.path.join(result_folder,test_data[0] + "_45_"+end_name,file)).columns.values.tolist()
    result_csv = os.path.join(deepcounting_folder, "plots_and_csv_statistics/" + file.split(".")[0]+ "_" +save_name +".csv")

    if os.path.exists(result_csv):
        df = pd.read_csv(result_csv)
    else:
        df = pd.DataFrame(columns=(["object","size"] + file_columns))

    lower_bound =  {"28":0, "45":28, "68":45,"384":68}

    for result_dir in os.listdir(result_folder):
        for object in test_data:
            for size in [28,45,68,384]:
                if object + "_" + str(size) in result_dir and result_dir.endswith(end_name):
                    temp_df = pd.read_csv(os.path.join(result_folder,result_dir,file))
                    if len(temp_df) != 33 and len(temp_df) != 11:
                        print(f"Warning! the folder {object}_{size} has processed only {len(temp_df)}/11 images")
                    temp_df["size"] = [size]*len(temp_df)
                    temp_df["object"] = [object]*len(temp_df)
                    temp_new_columns = ["object","size"]+file_columns
                    temp_df = temp_df[temp_new_columns]
                    df = pd.concat([df,temp_df],ignore_index=True)

    df.to_csv(result_csv, index=False)

def generate_ar_corr_amp(csv_file):
    test_data = ['Nails_23','Nails_32','Pins2','Nuts','Small_nails_23', 'Small_nails_32']
    data_df = pd.read_csv(csv_file)

    df = pd.DataFrame(columns=["size"] + test_data)

    lower_bound =  {"28":0, "45":28, "68":45,"384":68}

    for s in [28,45,68,384]:
        new_row = []
        for obj in test_data:
            #!r allows to have an object.__repr__() instead of object.__str__()
            new_row.append(round(float(np.average(db.sql('''
                                                        SELECT f1_score 
                                                            FROM 
                                                                data_df 
                                                            WHERE 
                                                                method = 'otm' 
                                                                AND object = {!r} 
                                                                AND size = {!r}
                                                        
            '''.format(obj,s)).fetchnumpy()["f1_score"])),2))

        df1 = pd.DataFrame([["["+str(lower_bound[str(s)])+","+str(s)+"]"] + new_row], columns=["size","Nails_23","Nails_32","Pins2","Nuts","Small_nails_23", "Small_nails_32"])
        df = pd.concat([df, df1], ignore_index=True)

    print(df)
    new_df = pd.DataFrame(columns=["size", "<2:3", "[2:3,3:2]" , "3:2<"])
    for index, row in df.iterrows():
        new_row = [round(np.average([row["Nails_23"],row["Small_nails_23"]]),2), round(np.average([row["Nuts"],row["Pins2"]]),2) , round(np.average([row["Nails_32"],row["Small_nails_32"]]),2)]
        df1 = pd.DataFrame([[row["size"]] + new_row], columns=["size", "<2:3", "[2:3,3:2]" , "3:2<"])
        new_df = pd.concat([new_df, df1], ignore_index=True)

    saving_name = str(Path(csv_file).stem).split(".")[0].replace("hm_","")
    print(saving_name)
    new_df = new_df.set_index('size')
    new_df.to_csv(os.path.join(rootDirectory,saving_name + ".csv"))

    # plot the correlation matrix
    import seaborn as sns

    ax = sns.heatmap(new_df, annot=True, cmap='viridis', fmt=".2f", annot_kws={"size":18})
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.savefig(os.path.join(rootDirectory,saving_name + ".pdf"))
    plt.show()

#generate_ar_corr_amp(file)
#create_big_csv("hm.csv")

if __name__ == "__main__":
    parser = parser.parse_args()
    result_folder = os.path.join(deepcounting_folder, parser.result_folder)
    create_big_csv(parser.file, parser.save_name, parser.end_name)

    if parser.heat_map:
        generate_ar_corr_amp(os.path.join(deepcounting_folder, "plots_and_csv_statistics/" + parser.file.split(".")[0]+ "_" + parser.save_name +".csv"))
    





