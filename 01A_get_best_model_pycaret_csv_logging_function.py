# %%

"""This uses PyCaret to get the best model. It can show many other things but we get the best model via best = compare_models().

To get the string of the model we use str(best) so that we can use eval() to create a model to pass to the get_model_metrics so that we can log mean of metrics.

https://github.com/pycaret/pycaret/blob/master/tutorials/Binary%20Classification%20Tutorial%20Level%20Beginner%20-%20%20CLF101.ipynb
"""

# %%

from numpy import random
import random
import uuid

import pandas as pd
from numpy import mean
from numpy import std
from datetime import date, datetime

from pycaret.classification import (
    setup,
    compare_models,
    create_model,
    pull,
    tune_model,
    evaluate_model,
    save_model,
    finalize_model,
    predict_model,
)
from pycaret.classification import *

# from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier

# from ml_fns.get_model_metrics import run_cross_val_score_metrics

from utils.logging.log_to_csv import save_results

print("========================")
print("")
print(">>> PLEASE RUN NEXT CELL -> LOAD FN")

# %%

# LOAD FN


def do_run(
    log_file="./db/pycaret_results.csv",
    dataset="./datasets/mlr.csv",
    features="ALL",
    sample=0.95,
    tuned=True,
):

    FILE = log_file
    DATASET = dataset
    FEATURES = features
    SAMPLE = sample
    TUNED = tuned
    SESSION_ID = 101  # Random Seed for PyCaret
    RUN_ID = str(uuid.uuid4())
    SETUP = f"{RUN_ID} {SESSION_ID} - use train_size=0.9 for train/validate"

    df = pd.read_csv(DATASET)

    df.head()
    print("========================")
    print(">>> ---- RUN DETAILS")
    print("RUN_ID:", RUN_ID)
    print("DATASET:", DATASET)
    print("FEATUES:", FEATURES)
    print("TRAIN/VALIDATE:", SAMPLE)
    print("")

    # SAVE SAMPLE & SETUP
    SAVE_MODEL = False
    ######################

    train_set = df.sample(frac=SAMPLE)

    test_set = df.drop(train_set.index)
    # test_set is acually HOLDOUT set never seen by
    test_set.head()

    print("Data for modeling train and validate sets: " + str(train_set.shape))
    print("Unseen HOLDOUT Data For Predictions: " + str(test_set.shape))

    print("========================")
    print("")
    print(">>> PLEASE RUN NEXT CELL -> PROCESS")

    # PROCESS

    # train_size is ratio train/validate
    #  session_id: random_state
    # If we want to recreate PyCaret then use same session_id
    # s = setup(train_set, target='target',
    #           train_size=0.9, session_id=SESSION_ID)

    # Let PyCaret pick randon seed...
    s = setup(train_set, target="target", train_size=0.9)

    ######################

    # compare_models

    # This function trains and evaluates the performance of all estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the get_metrics function. Custom metrics can be added or removed using add_metric and remove_metric function.

    print(">>> Comparing models...")
    print("")
    print(
        "This function trains and evaluates the performance of all estimators available in the model library using cross-validation. The output of this function is a scoring grid with average cross-validated scores. Metrics evaluated during CV can be accessed using the pull() function. Custom metrics can be added or removed using add_metric and remove_metric function."
    )

    # best = compare_models(include=["ada", "xgboost", "gbc", "rf", "lr", "et"])
    best = compare_models(include=["rf"])

    # best = compare_models()

    ######################

    #  to run eval on the best model we need to ensure it is a string as best is reference to a class.
    str_best = str(best)
    print("BEST model and hyperparameters are:")
    print(best)
    print("================================")
    print(
        "Using best model, create a model and rerun to be able to pull out the dataframe from"
    )

    lr = create_model(best)
    if TUNED:
        print("!!!!!!!!!!!!!")
        print("....tuning...")
        lr = tune_model(lr)
        print("TUNING APPLIED")

    pull_metrics = pull()
    print("PULL metrics from above TUNED dataframe...")
    print("MEAN Accuracy", pull_metrics["Accuracy"].loc["Mean"])
    print("MEAN AUC", pull_metrics["AUC"].loc["Mean"])
    print("MEAN Recall", pull_metrics["Recall"].loc["Mean"])
    print("MEAN Prec.", pull_metrics["Prec."].loc["Mean"])
    print("MEAN F1", pull_metrics["F1"].loc["Mean"])
    print("MEAN Kappa", pull_metrics["Kappa"].loc["Mean"])
    print("MEAN MCC", pull_metrics["MCC"].loc["Mean"])

    print("========================")
    print("")
    print(">>> PLEASE RUN NEXT CELL -> FINALIZE")

    # FINALIZE_MODEL

    # This function trains a given model on the entire dataset including the hold-out set.

    print(">>> Finalizing model...")
    print("")
    print(
        "This function trains a given model on the entire dataset including the hold-out set."
    )

    finalize_model(best)

    print("========================")
    print("")
    print(">>> PLEASE RUN NEXT CELL -> EVALUATE")

    # EVALUATE
    if SAVE_MODEL:
        # SAVE THE MODEL
        print(">>> SAVING best model")
        save_model(best, f"{RUN_ID}")

    # EVALUATE_MODEL(best)
    # This function analyzes the performance of a trained model on the test set. It may require re-training the model in certain cases.

    print(">>> Evaluating model...")
    print(
        "This function analyzes the performance of a trained model on the test set. It may require re-training the model in certain cases."
    )
    print("")
    evaluate_model(best)

    # tune_model(best, optimize='AUC')

    # plot_model(lr, plot='confusion_matrix', plot_kwargs={'percent': True})

    print("========================")
    print("")
    print(">>> PLEASE RUN NEXT CELL -> PREDICT")

    # GET METRICS

    # print('================================')
    # print('Using best model, create a model and rerun to be able to pull out the dataframe from')

    # print('========================')
    # print('')
    # print('>>> PLEASE RUN NEXT CELL -> PREDICT')

    # PREDICT MODEL

    print("================================")
    print("predict_model(best, test_set=HOLDOUT_SET)")
    print(
        "This function predicts Label and Score (probability of predicted class) using a trained model. When data is None, it predicts label and score on the holdout set."
    )
    print("")
    preds = predict_model(best, test_set)
    #  prob score available but not used as it differs for models and may not be present - use try/except?
    results = preds[["target", "prediction_label"]]

    results["target"] = results["target"].replace({True: 1, False: 0})
    results["correct"] = results["target"] == results["prediction_label"]
    print("================================")
    print("HOLDOUT RESULTS DATAFRAME")
    print(pd.DataFrame(results))
    pd.DataFrame(results)
    print("========================")
    print("")

    ######################
    # SAVE holdout_correct

    holdout_correct = round(results["correct"].sum() / len(results["correct"]), 2)
    ######################
    print("================================")
    print("Logging to CSV...")
    print(
        round(results["correct"].sum() / len(results["correct"]), 2) * 100,
        "%",
        "correct on => preds = predict_model(best, test_set) dataset with test_set being the HOLDOUT/FUTURE set",
    )
    results["correct"] = results["correct"].replace({True: "YES", False: "NO"})
    results[["target", "prediction_label", "correct"]]

    print("========================")
    print("")
    print(">>> PLEASE RUN NEXT CELL -> LOGGING")

    # LOGGING

    lr = create_model(best)
    # dashboard(lr)

    pull_metrics = pull()
    print("PULL metrics from above dataframe...")
    print("MEAN Accuracy", pull_metrics["Accuracy"].loc["Mean"])
    print("MEAN AUC", pull_metrics["AUC"].loc["Mean"])
    print("MEAN Recall", pull_metrics["Recall"].loc["Mean"])
    print("MEAN Prec.", pull_metrics["Prec."].loc["Mean"])
    print("MEAN F1", pull_metrics["F1"].loc["Mean"])
    print("MEAN Kappa", pull_metrics["Kappa"].loc["Mean"])
    print("MEAN MCC", pull_metrics["MCC"].loc["Mean"])
    print("MEAN F1", pull_metrics["F1"].loc["Mean"])

    mean_metrics = {
        "accuracy": pull_metrics["Accuracy"].loc["Mean"],
        "roc_auc": pull_metrics["AUC"].loc["Mean"],
        "recall": pull_metrics["Recall"].loc["Mean"],
        "precision": pull_metrics["Prec."].loc["Mean"],
        "f1": pull_metrics["F1"].loc["Mean"],
        "kappa": pull_metrics["Kappa"].loc["Mean"],
        "mcc": pull_metrics["MCC"].loc["Mean"],
    }
    RUN_DATE = date.today()
    # run_id,run_date,mlr_dataset, feature_set,split,tuned,setup,best,pred_accuracy, metrics_dict, accuracy, roc_auc, recall, precision, f1,kappa,mcc
    project_id = 1
    data_scientist_id = 1

    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M")
    DATA = [
        RUN_ID,
        date_time_str,
        project_id,
        data_scientist_id,
        DATASET,
        FEATURES,
        SAMPLE,
        TUNED,
        SETUP,
        best,
        holdout_correct,
        mean_metrics,
        mean_metrics["accuracy"],
        mean_metrics["roc_auc"],
        mean_metrics["recall"],
        mean_metrics["precision"],
        mean_metrics["f1"],
        pull_metrics["Kappa"].loc["Mean"],
        pull_metrics["MCC"].loc["Mean"],
    ]

    print("RUN_ID:", RUN_ID)
    print("================================")
    print("saving RUN DETAILS to CSV", date_time_str)
    print("================================")

    # 0.7109	0.7326	0.8738	0.7887	0.7476

    save_results(FILE, DATA)

    print("***************************")
    print(">>> RUN ENDED")


print("========================")
print("")
print(">>> PLEASE RUN NEXT CELL -> BATCH")

# %%

# BATCH

count = 0
runs = []
path = "./datasets/"
#  create random sample size with bias towards 0.90+
lst_sample_size = [0.90, 0.95]
lst_datasets = [
    "mlr.csv",
    "mlr.csv",
]


for i in range(0, len(lst_datasets)):
    random_sample_size = lst_sample_size[random.randint(0, len(lst_sample_size)) - 1]
    run_dataset = path + lst_datasets[i]
    run_data = f"{run_dataset} {random_sample_size}"
    runs.append(run_data)
    rand_tuned = True
    try:
        do_run(
            dataset=run_dataset,
            features="ALL",
            sample=random_sample_size,
            tuned=rand_tuned,
        )

        count += 1
        print("+++++++++++++++++++++")
        print("RUN NUMBER:", count)
        print("+++++++++++++++++++++")

    except:
        pass

print("+++++++++++++++++++++++++++")
print("")
print(f"{count} runs completed")
for item in runs:
    print(item)

# %%
