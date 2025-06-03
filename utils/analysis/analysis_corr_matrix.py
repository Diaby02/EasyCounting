import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"c:\Users\bourezn\Documents\Master_thesis\utils\input_patch_k32.csv")

only_kernel_3 = True
if only_kernel_3:
    df2 = pd.read_csv(r"c:\Users\bourezn\Documents\Master_thesis\utils\input_patch_human.csv")

    merge_df = df.merge(df2,right_on=["image_name","resized_size","patch_size","Avg_ref_h","Avg_ref_z","mean_box_size","diff_patch_mbs"], left_on=["image_name","resized_size","patch_size","Avg_ref_h","Avg_ref_z","mean_box_size","diff_patch_mbs"])
    merge_df = merge_df.drop(columns=["mae_x"])
    merge_df = merge_df.rename(columns={'mae_y': 'mae'})
else:
    merge_df = df

number_of_columns = len(merge_df.columns.values)

diff_c = df["diff_patch_mbs"]
mae_c  = df["mae"]

correlation = diff_c.corr(mae_c)
print(correlation)

# coorelation matrix between the 6 last columns
interesting_columns = ["patch_size","mean_box_size","diff_patch_mbs","aspect_ratio","std_size","mae","mre","score","qual_score"]

if only_kernel_3:
    merge_df["score"] = round(np.ones(len(merge_df)) - merge_df["score"],2) # change such that how small the score is, how good the map is
    merge_df.to_csv("merge_comparison_f1.csv")

correlation_matrix = merge_df[interesting_columns].corr(method="kendall") # kendall is used for non-linear relationships, assume monotonic relationships

correlation_matrix_tiny = correlation_matrix[["patch_size","mean_box_size","diff_patch_mbs","aspect_ratio","std_size"]]
correlation_matrix_tiny = correlation_matrix_tiny.iloc[5:]

# plot the correlation matrix
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.tight_layout()
plt.savefig("plots/correlation_map_tiny.pdf")
plt.show()

# plot the data
plt.scatter(diff_c, mae_c)
plt.xlabel('diff_patch_mbs')
plt.ylabel('mae')
plt.title('diff_patch_mbs vs mae')

# add a trendline
import numpy as np
m, b = np.polyfit(diff_c, mae_c, 1)
plt.plot(diff_c, m*diff_c + b)
plt.show()


"""

- entrainer les modèles avec des kernels 5x5 et 7x7
- juger la qualité discriminative des modèles selon un score qualitatif allant de 0 à 1, par tranche de 0.1
- étudier l'aspect ratio des bounding boxes et ajouter le critère, et voir l'influence sur la qualité des modèles
- étudier le rapport entre la taille moyenne de la bouding box et la taille du patch en entrée, et voir l'influence sur la qualité du modèle
- voir l'influence du rapport image_size/patch et feature_size/kernel_size sur la qualité du modèle
"""

# statistical analysis

import statsmodels.api as sm




