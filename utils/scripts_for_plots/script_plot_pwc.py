from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import matplotlib.dates as mdates
    
    
dict_models_date = {"GeCO": "2024-12-15","CountDiff":"2024-12-01", 
                "CountGD": "2024-08-01","DAVE": "2024-07-15",
                "VA-Count":"2024-07-01","SSD":"2024-05-20", 
                "PseCo":"2024-04-01", "Omnicount":"2024-03-01",
                "CounTX":"2023-08-01","CACViT":"2023-07-01",
                "CLIP-Count":"2023-05-01","SPDCN":"2023-12-01",
                "LOCA":"2022-11-01","RCAC":"2022-10-01",
                "RCC":"2022-09-01","Counting-DETR":"2022-08-01",
                "RCC":"2022-07-01","CounTR": "2022-06-15",
                "BMNet":"2022-06-01", "VCN":"2022-05-01",
                "SAFECount":"2022-01-01","LaoNet":"2021-12-01",
                "FamNet":"2021-06-01", "CFOCNet":"2021-01-01"}

dict_models_test = {"GeCO": 7.91,"CountDiff":9.24, 
                "CountGD": 5.74,"DAVE": 8.66,
                "VA-Count":17.88,"SSD":9.58, 
                "PseCo": 13.05, "Omnicount":18.63,
                "CounTX":15.88,"CACViT":9.13,
                "CLIP-Count":17.78,"SPDCN": 13.51,
                "LOCA":10.79,"RCAC":20.21, "Counting-DETR": 16.79,
                "RCC":17.12,"CounTR": 11.95,
                "BMNet":14.62, "VCN":18.17,
                "SAFECount": 14.32,"LaoNet": 15.78,
                "FamNet":22.08, "CFOCNet":22.10}

dates = [datetime.strptime(d, "%Y-%m-%d") for d in dict_models_date.values()]
maes =  dict_models_test.values()
models = dict_models_date.keys()
dates, models, mae = zip(*sorted(zip(dates,models, maes)))


x_labels = dates
y_val = maes

fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")

best_maes_trough_time = []
best_date_trough_time = []
best_mae = 100
for model in models:
    current_mae = dict_models_test[model]
    current_date = dict_models_date[model]
    current_date = datetime.strptime(current_date, "%Y-%m-%d")
    if current_mae < best_mae:
        best_maes_trough_time.append(current_mae)
        best_date_trough_time.append(current_date)
        best_mae = current_mae

#x=list(df[DATE_COLUMN].astype('datetime64[us]').values)

for (date, model) in zip(dates,models):
    mae = dict_models_test[model]
    if dict_models_test[model] in best_maes_trough_time:
        ax.scatter(date, mae, color="blue", s=25, zorder=2)
        ax.annotate(str(model), (date, mae+(max(dict_models_test.values())/25)), color="blue")
    else:
        ax.scatter(date, mae, color="grey", s=25, zorder=2)
        ax.annotate(str(model), (date, mae+(max(dict_models_test.values())/25)), color="grey", size=8)

ax.plot(best_date_trough_time, best_maes_trough_time, "ko",color="blue", linestyle= "-", linewidth=1.5, zorder=1)
ax.xaxis.set(major_locator=mdates.YearLocator(), major_formatter=mdates.DateFormatter("%Y"))

ax.margins(0.06,0.1)
ax.set_ylabel("MAE")
ax.set_xlabel("Time")
#ax.set_title("MAE of the selected FSC-models on the FSC147-test dataset")

plt.savefig("plots/FSC147_test_all_models.pdf")
plt.show()