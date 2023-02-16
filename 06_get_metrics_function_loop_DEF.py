# %%
"""
Uses a dataset to do train/test but uses ANOTHER dataset as test.

test_size=0.1 (same model metrics as for 0.15)so that train size is a large as possible as this is a routine that does not load in an existing model...TO DO.

model gives 0.88 accuracy.


"""

# %%

import random
from datetime import datetime
import uuid
from csv import writer
import numpy as np
import pandas as pd
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


def do_loop():

    lst_split = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    SPLIT = lst_split[random.randint(0, len(lst_split)-1)]
    # SPLIT = 0.15
    SPLIT

    if random.randint(1, 100) > 50:
        DO_SCALING = True
    else:
        DO_SCALING = False

    # DO_SCALING = False
    DO_SCALING

    x = random.randint(1, 300)
    if x < 100:
        MODEL_TYPE = 'log'
    elif x < 200:
        MODEL_TYPE = 'ran'
    else:
        MODEL_TYPE = 'ada'

    # MODEL_TYPE = 'ada'

    MODEL_TYPE

    # For train/test/split
    RND_NUM_SPLIT = random.randint(1, 1000)
    # RND_NUM = random.randint(1120,1140)

    RND_NUM_SPLIT
    # For model hyperparameters
    RND_NUM_HYPER = random.randint(1, 10000)

    RND_NUM_HYPER

    lst_datasets = [

        './datasets/reduced_features_mlr_datasets/RED_ADA_RANKED_101.csv',

        './datasets/reduced_features_mlr_datasets/KEEP_RED_MLR_BEST_f2f88ce7-7be6-47c6-abce-e1a2697ac866.csv',
        './datasets/mlr_spearman.csv',
    ]
    len(lst_datasets)

    DATASET = lst_datasets[random.randint(0, len(lst_datasets)-1)]
    DATASET

    df = pd.read_csv(DATASET)

    # df.head()

    # sns.countplot(data=df, x='target')

    # Model evaluation
    X = df.drop('target', axis=1)
    y = df['target']
    X.shape

    X
    #  Model

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPLIT, random_state=RND_NUM_SPLIT)
    # try random_state=3 original is 101

    if DO_SCALING:
        try:
            scaler = StandardScaler()
            # scaling does not seem to affect results
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        except:
            print('SCALING FAIL')

    # 1113

    if MODEL_TYPE == 'ada':
        model = AdaBoostClassifier(
            algorithm='SAMME.R', learning_rate=1.0, n_estimators=50, random_state=RND_NUM_HYPER)

    if MODEL_TYPE == 'log':
        model = LogisticRegression(max_iter=1000)

    if MODEL_TYPE == 'ran':
        model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                       criterion='gini', max_depth=None, max_features='sqrt',
                                       max_leaf_nodes=None, max_samples=None,
                                       min_impurity_decrease=0.0, min_samples_leaf=1,
                                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                                       n_estimators=100, n_jobs=-1, oob_score=False,
                                       random_state=RND_NUM_HYPER, verbose=0, warm_start=False)

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
    holdout_correct = df_results['correct'].sum()/len(df_results)
    df_results['correct'] = df_results['correct'].replace(
        {True: 'YES', False: 'NO'})
    df_results

    holdout_correct = round(holdout_correct, 2)
    holdout_correct

    # Evaluate MODEL

    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions, )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    #  LOG DATA

    timestamp = datetime.now()
    FILE = './test_results_loop_def.csv'

    RUN_ID = str(uuid.uuid4())

    DATA = [RUN_ID, timestamp, DATASET, MODEL_TYPE, RND_NUM_HYPER,
            SPLIT, RND_NUM_SPLIT, DO_SCALING, holdout_correct]

    save_results(FILE, DATA)
    print('DATASET', DATASET)
    print('FILE', FILE)


# %%

for run in range(100):
    do_loop()
    print(run)


print('')
print('')
print('++++++++++++++++++++++++++')
print('LOOP FINISHED')
# %%
