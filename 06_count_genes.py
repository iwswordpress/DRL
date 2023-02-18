# %%
import pandas as pd

df_rnd_used = pd.read_csv("./csvs/gene_importances/rnd_used.csv")
total = len(df_rnd_used)
print("total number of runs", total)
df = pd.read_csv("./csvs/gene_importances/genes.csv")
# df
df_count = df.value_counts("gene")
print("GENE OCCURRENCE")
print(df_count)
df_count.to_csv("./csvs/gene_importances/gene_count.csv")

# %%
