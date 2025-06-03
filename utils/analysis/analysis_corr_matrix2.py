import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"c:\Users\bourezn\Documents\Master_thesis\utils\merge_comparison_f1.csv")

interesting_columns = ["mae","mre","score","qual_score","f1_otm","f1_gmns","f1_hdbscan"]
correlation_matrix = df[interesting_columns].corr(method="kendall") # kendall is used for non-linear relationships, assume monotonic relationships

correlation_matrix_tiny = correlation_matrix[ ["mae","mre","score","qual_score","f1_otm","f1_gmns","f1_hdbscan"]]
correlation_matrix_tiny = correlation_matrix_tiny.iloc[-3:]

# plot the correlation matrix
import seaborn as sns
sns.heatmap(correlation_matrix_tiny, annot=True, cmap='viridis', fmt=".2f")
plt.tight_layout()
plt.savefig(r"analysis_datasets_results\merge_comparison_f1.pdf")
plt.show()

df_comp = pd.read_csv(r"c:\Users\bourezn\Documents\Master_thesis\utils\input_patch_comparison.csv")
df_comp.head()