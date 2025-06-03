from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import matplotlib.dates as mdates
    
    
dict_models = {"GeCO": "2024-12-15","CountDiff":"2024-12-01", 
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

#                "GMN": "2018-11-01","MAML":"2017-03-01"}

dates = [datetime.strptime(d, "%Y-%m-%d") for d in dict_models.values()]
models = [model for model in dict_models.keys()]
dates, models = zip(*sorted(zip(dates,models)))

# Choose some nice levels: alternate meso releases between top and bottom, and
# progressively shorten the stems for micro releases.
levels = []
down = [i + j for i in range(3,len(models),6) for j in range(3)]
up = [i + j for i in range(0,len(models),6) for j in range(3)]

print(up)
print(down)
for i,model in enumerate(dict_models):

    if i >= 0 and i <= 3:
        h = 2.3
        level = h if i % 2 == 0 else -h
    else:
        h = 0.5 + 0.6 * (3 - (i % 3))

        if i in down:
            level = h 
        else:
            level = -h

    levels.append(level)



# The figure and the axes.
fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
#ax.set(title="Date of publication of the object class-agnostic counting models since 2021")

# The vertical stems.
colors = ["tab:blue"]*len(models)
colors[1] = "tab:green"
colors[20] = "tab:red"

ax.vlines(dates, 0, levels,
          color=colors)
# The baseline.
ax.axhline(0, c="black")
# The markers on the baseline.
ax.plot(dates, np.zeros_like(dates), "ko", mfc="white") 
ax.plot([dates[1]], np.zeros_like([dates[1]]), "ko", mfc="tab:green")
ax.plot([dates[20]], np.zeros_like([dates[20]]), "ko", mfc="tab:red")

# Annotate the lines.
for date, level, model in zip(dates, levels, models):
    version_str = model
    ax.annotate(version_str, xy=(date, level),
                xytext=(-3, np.sign(level)*3), textcoords="offset points",
                verticalalignment="bottom" if level > 0 else "top",
                weight="bold" if model in ["FamNet", "CountGD"] else "normal",
                bbox=dict(boxstyle='square', pad=0, lw=0, fc=(1, 1, 1, 0.7)))

ax.xaxis.set(major_locator=mdates.YearLocator(),
             major_formatter=mdates.DateFormatter("%Y"))

# Remove the y-axis and some spines.
ax.yaxis.set_visible(False)
ax.spines[["left", "top", "right"]].set_visible(False)

ax.margins(y=0.1)

green_patch = mpatches.Patch(color="green", label="Baseline model")
blue_patch = mpatches.Patch(color="red", label="SOTA")

ax.legend(handles=[green_patch,blue_patch],loc="lower left")

plt.savefig("plots/models_time_line_bigger.pdf")
plt.show()