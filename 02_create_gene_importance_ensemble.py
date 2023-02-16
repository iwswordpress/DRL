# %%

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings("ignore")

""" Create CSV of gene importortance for Ensemble Methods.

File stored in csvs folder.

Has different internal coeffs to log reg method so is separate function.

It is possible to refactor both to one function...

"""


def create_ensemble_gene_importance(
    model,
    model_type,
    dataset,
    do_scaling=True,
    do_stratify=True,
    split=0.15,
    rnd_state_hyper=4098,
    rnd_state_split=101,
):
    # RUN_ID = str(uuid.uuid4()) replace
    # HYPERPARAMETER_RUN_ID links to pycaret RUN_ID from where we got hyperparameters
    # HYPERPARAMETER_RUN_ID = rn_state
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
    type(importance)
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
    df_ensemble_rated.sample()
    output_filename = f"./csvs/gene_importances/{model_type.upper()}_RANKED_{rnd_state_hyper}_{rnd_state_split}.csv"
    print(output_filename)
    df_ensemble_rated.to_csv(output_filename, index=False)

    return


# %%
RND_STATE_HYPER = 1113
RND_STATE_SPLIT = 101

# %%
model = AdaBoostClassifier(
    algorithm="SAMME.R",
    learning_rate=1.0,
    n_estimators=50,
    random_state=RND_STATE_HYPER,
)
create_ensemble_gene_importance(
    model,
    "ada",
    dataset="RED_ADA_RANKED_101.csv",
    do_scaling=True,
    do_stratify=True,
    split=0.10,
    rnd_state_hyper=RND_STATE_HYPER,
    rnd_state_split=RND_STATE_SPLIT,
)

# %%
model = RandomForestClassifier(
    bootstrap=True,
    ccp_alpha=0.0,
    class_weight=None,
    criterion="gini",
    max_depth=None,
    max_features="sqrt",
    max_leaf_nodes=None,
    max_samples=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=1,
    min_samples_split=2,
    min_weight_fraction_leaf=0.0,
    n_estimators=100,
    n_jobs=-1,
    oob_score=False,
    random_state=RND_STATE_HYPER,
    verbose=0,
    warm_start=False,
)
create_ensemble_gene_importance(
    model,
    "ran",
    dataset="RED_ADA_RANKED_101.csv",
    do_scaling=True,
    do_stratify=True,
    split=0.10,
    rnd_state_hyper=RND_STATE_HYPER,
    rnd_state_splitr=RND_STATE_SPLIT,
)
