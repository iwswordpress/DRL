![Explorer](./_images/explorer.png)

## STEP 01: START - get best hyperparameters and model

Start with <font size="3">**01_get_best_model_pycaret_csv_logging_function.py**</font> which takes a dataset and can be run in loop.

This runs models, produces pycaret's 'best' and we log all the runs in pycaret_results.csv in db:

run_id,run_date,mlr_dataset,split,setup,best,pred_accuracy,metrics_dict, accuracy,precision,recall,f1,roc_auc

## STEP 02: FEATURE IMPORTANCES - create many importances for LogReg/Ensemble methods.

This is for ensenble types as the method to get fetaure importances is different to say LogReg.

They could be refactored into one file as the difference is the fact that log_reg has a different attribute for importances. Being LogReg it also has direction which ensemble models do not have.

We can make the assumption, prior to say SHAP, can give direction of genes for ensemble models.

These csvs of gene importance for LogReg and Ensemble models are stores in <font size="3">**_csvs/gene_importances_**</font>

We can run the modelling again on a reduced feature set based on the importances or use XAI as the featureset is much smaller.

## STEP 03: CREATE REDUCED MLR DATASETS

<font size="3">**03_create_filtered_mlr_dataset.ipynb**</font> will create an MLR with selected features based on a filter applied to merged importances set. This will be stored in <font size="3">**_datasets/reduced_features_mlr_datasets_**</font>.

## STEP 04: METRICS AND PLOTS FOR MODEL AND AGAINST OTHER

<font size="3">**04_get_metrics_versus_OTHER.ipynb**</font> redoes metrics but adds in some CM plots. It alse tests model against OTHER.

<font size="3">**05_get_best_model_pycaret_csv_logging_interactive_OTHER_HOLDOUT.py**</font> is a single run version of <font size="3">**01_get_best_model_pycaret_csv_logging_function.py**</font> with option to use OTHER as HOLDOUT.

## REDO STEPS: RERUN MODELLING

We can then repeat the loop...

Using <font size="3">**get_best_model_pycaret_csv_logging_interactive_OTHER_HOLDOUT.py**</font> we use the OTHER dataset of 123 records as the holdout set. We need to filter the columns to match that of the MLR train set.

Results are logged to a CSV file for each run as <font size="3">pycaret_holdout_results_5dc322da-8454-4f29-ba3a-6ee22aec7b2a.csv</font> and it can be cross referenced with the RUN_ID once I have added back the logging of run data. TODO!!!

## 06FEB2023

<font size="3">**get_best_model_pycaret_csv_logging_function.ipynb**</font> now uses tuning and is set up as a function so GridSearch can be done.

<font size="3">**pycaret_results_all_06JAN2023_KEEP.xlsx**</font> is one good set of runs.

<font size="3">**html_notebooks/Ada_optimum_v_OTHER.html**</font> has the best so far model which was Ada and accuracy 0.88 and trains a model on this with test split of 0.1 which is optimum.

Then the OTHER is evaluated against it but gives around 0.5

Using the ADA best model we find the gene importance gives 50 non-zero genes. The hyperparameters had estimators of 50 so this might be why this number of non-zero is 48(???) Tested with 100 estimators and gave 87 non-zero so just coincidence but a factor.

Running this reduced set through PyCaret gives better metrics:

0.79 0.83 0.85 0.84 0.83 0.56 0.59 - all features

0.83 0.90 0.84 0.88 0.85 0.65 0.67 - same model and hyperparameters

came up with same hyperparameters and model as best but less noise...

"AdaBoostClassifier(algorithm='SAMME.R', , learning_rate=1.0,
n_estimators=50, random_state=6301)"

base_estimator=None now deprecated

<font size="3">**RED_MLR_BEST_f2f88ce7-7be6-47c6-abce-e1a2697ac866.csv**</font> has two agenes at 0.04 the other 46 at 0.02.

RUN <font size="3">**04_get_metrics_versus_OTHER.ipynb**</font> with various splits, datasets, optimised set and even just the - full, top and just top two...DO each separately and then feature addition FSE...
