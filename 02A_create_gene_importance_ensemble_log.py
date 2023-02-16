# %%

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import uuid
import warnings
import random
from datetime import date, datetime
import csv
from functools import reduce

from utils.logging.log_to_csv import save_results

warnings.filterwarnings("ignore")

""" Create CSV of gene importortance for Ensemble Methods.

File stored in csvs folder.

Has different internal coeffs to log reg method so is separate function.

It is possible to refactor both to one function...

"""


def create_ensemble_gene_importance(
    model,
    model_type="ADA",
    do_scaling=True,
    do_stratify=True,
    split=0.10,
    rnd_state_split=101,
    rnd_state_hyper=1,
    dataset="mlr_spearman.csv",
):
    print(model_type, rnd_state_hyper)
    model_type = model_type.lower()
    print(model_type)
    key_imp = f"{model_type}_imp"
    key_rank = f"{model_type}_rank"

    df = pd.read_csv(f"./datasets/{dataset}")
    # shuffle dataset
    df = df.sample(frac=1)
    df.head()

    X = df.drop("target", axis=1)
    y = df["target"]
    y.value_counts()

    if do_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=rnd_state_split, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=rnd_state_split
        )
    cols_features = X_train.columns

    scaler = StandardScaler()

    if do_scaling:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    def get_direction(row):
        if row["importance"] > 0:
            return "positive"
        elif row["importance"] < 0:
            return "negative"
        else:
            return "NIL"

    model.fit(X_train, y_train)

    importance = model.feature_importances_
    # type(importance)

    df_importance = pd.DataFrame(importance, index=cols_features)
    df_importance = df_importance.rename(columns={0: "importance"})
    df_importance["direction"] = df_importance.apply(
        lambda row: get_direction(row), axis=1
    )
    df_importance.reset_index(inplace=True)

    df_importance = df_importance.rename(
        columns={"index": "gene", "importance": key_imp}
    )

    df_importance = df_importance.rename(columns={"index": "gene"})

    df_ensemble_rated = df_importance.sort_values(key_imp, ascending=False)

    df_ensemble_rated.sample()

    seq = list(range(len(df_ensemble_rated)))
    df_ensemble_rated[key_rank] = seq
    df_ensemble_rated[key_rank] = df_ensemble_rated[key_rank] + 1

    return df_ensemble_rated


# %%

#  load rnd_used.csv and convert to list

file = open("rnd_used_003.csv", "r")
rnd_used = list(csv.reader(file, delimiter=","))
file.close()
rnd_used = reduce(lambda x, y: x + y, rnd_used)
rnd_used = list(filter(lambda a: a != "0", rnd_used))
print("rnd used so far...")
print(rnd_used)
print("================")
RUN_ID = str(uuid.uuid4())
for _ in range(5000):
    RND_STATE_HYPER = random.randint(10_000_000,300_000_000)
    if RND_STATE_HYPER in rnd_used:
        continue
    rnd_used.append(RND_STATE_HYPER)

    model = AdaBoostClassifier(
        algorithm="SAMME.R",
        learning_rate=1.0,
        n_estimators=75,
        random_state=RND_STATE_HYPER,
    )
    df = create_ensemble_gene_importance(
        model,
        rnd_state_hyper=RND_STATE_HYPER,
    )
    df_non_zero = df[df["ada_imp"] > 0]

    df_non_zero
    df_non_zero["run_id"] = str(uuid.uuid4())
    df_non_zero["rnd_state_hyper"] = RND_STATE_HYPER
    df_non_zero[
        "model"
    ] = "AdaBoostClassifier(algorithm='SAMME.R',learning_rate=1.0,n_estimators=75,random_state=RND_STATE_HYPER"
    df_non_zero["model_type"] = "ADA"
    df_non_zero
    try:
        df_non_zero.to_csv("genes_003.csv", mode="a", index=False, header=False)
    except:
        pass
    pd_rnd_used = pd.Series(rnd_used)
    pd_rnd_used.to_csv("rnd_used_003.csv", index=False)

print("finished...")


# %%
