# %%
import pandas as pd

df_rnd_used = pd.read_csv("./rnd_used_003.csv")
total = len(df_rnd_used)
print("total number of runs", total)
df = pd.read_csv("./genes_003.csv")
# df
df_count = df.value_counts("gene")
print("GENE OCCURRENCE")
print(df_count)
df_count.to_csv("gene_count_003.csv")

# %%
