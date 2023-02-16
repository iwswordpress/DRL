# %%
"""
We run metrics using the Ada best model but we can pass in the dataset, random_state for train/test/split and also for the model hyperparameter, as well as whether to use scaling.

Datasets can be constructed from feature importances made with 02_create_gene_importance, queried in some way and then use the resulting list of genes to filter out all non-included genes from the main dataset.]

This is how RED_ADA_RANKED_101.csv dataset was made using 03_create_filtered_mlr_dataset.ipynb


"""

# %%
import numpy as np
import pandas as pd
from datetime import datetime
import random
import uuid
from csv import writer

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.logging.log_to_csv import save_results

# %%


def save_test_results(file, data):
    with open(file, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(data)
        f_object.close()


def run_metrics_test(model='ada', dataset='./datasets/mlr_ada_2g.csv', split=0.1, rand_train_test_split=101, rand_hyper=1113, do_scaler=True):

    df = pd.read_csv(dataset)

    df.head()

    # sns.countplot(data=df, x='target')

    # MODEL

    # Model evaluation
    X = df.drop('target', axis=1)
    y = df['target']
    X.shape

    #  if using mlr_ada_best.csv which has 48 non-zero feature importances from ADA best set of hyperparameters: AdaBoostClassifier(algorithm='SAMME.R', base_estimator='deprecated', learning_rate=1.0,n_estimators=50, random_state=1113)

    # Selecting top two give 0.88
    # START = 0
    # END = 2
    # Select first N columns
    # X = X.iloc[:, START:END]

    X

    #  Model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=rand_train_test_split)
    # try random_state=3 original is 101
    if do_scaler:
        scaler = StandardScaler()
        # scaling does not seem to affect results
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if model == 'ada':
        model = AdaBoostClassifier(algorithm='SAMME.R', 
                                   learning_rate=1.0, n_estimators=50, random_state=rand_hyper)
    if model == 'log':
        model = LogisticRegression(max_iter=1000)
    
    if model == 'ran':
        model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=100, n_jobs=-1, oob_score=False,
                       random_state=4098, verbose=0, warm_start=False)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    predictions

    df_predictions = pd.DataFrame(predictions)
    df_predictions.rename({0: 'pred'}, axis=1, inplace=True)
    # df_predictions

    df_y_test = pd.DataFrame(y_test)
    df_y_test.rename({'target': 'actual'}, axis=1, inplace=True)
    # df_y_test

    # Combine two df with different inexes
    df_preds = pd.concat([df_y_test, df_predictions], axis=0)
    df_preds
    df_results = pd.concat([df_y_test.reset_index(
        drop=True), df_predictions.reset_index(drop=True)], axis=1)
    df_results['correct'] = df_results['actual'] == df_results['pred']
    holdout_correct = round(
        df_results['correct'].sum()/len(df_results['correct']), 2)
    print('HOLDOUT CORRECT:', holdout_correct)

    df_results

    ######################
    print('================================')
    print('Logging to CSV...')

    # Evaluate MODEL

    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions, )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    timestamp = datetime.now()
    FILE = './test_results.csv'

    RUN_ID = str(uuid.uuid4())

    DATA = [RUN_ID, timestamp, dataset, model,
            split, do_scaler, rand_train_test_split, holdout_correct]

    print('RUN_ID:', RUN_ID)
    print('================================')
    print('saving RUN DETAILS to CSV', timestamp)
    print('DATASET', dataset)
    print('HOLDOUT CORRECT:', holdout_correct)
    print('DATA:', DATA)
    print('================================')
    save_test_results(FILE, DATA)
    print('end csv logger function save_results()')
    return




# %%

# run_metrics_test()

# 0.2 101  5555  -> 0.79
# 0.1 101  5555  -> 0.94
# 0.1 101  5555  -> 0.82
# 0.1 5555 5555  -> 0.65
# 0.1 101  22    -> 0.82
# 0.2 101  22    -> 0.67
# 0.2 3    22    -> 0.67
# 0.1 3    22    -> 0.82
run_metrics_test(model='ada', dataset='./datasets/mlr_ada_best.csv',
                 split=0.1, rand_train_test_split=101, rand_hyper=5101, do_scaler=True)

print('')
print('')
print('========== RED_RAN_RANKED_101.csv ==========')
print('')
print('')
run_metrics_test(model='ran', dataset='./datasets/RED_RAN_RANKED_101.csv',
                 split=0.1, rand_train_test_split=101, rand_hyper=5555, do_scaler=True)


# %%

#  This is the 'best' model/hyperparameters from PyCaret, although a better RanForest has subsequently been found.

#  The rand we pass in is for train/test/split random_state so we would need to change the random_state for the model manually inside the function.

run_metrics_test(model='ada', dataset='./datasets/RED_ADA_RANKED_101.csv',
                 split=0.1, rand_train_test_split=5, rand_hyper=1113, do_scaler=True)

# %%
