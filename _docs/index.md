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

## STEP 04: METRICS AND PLOTS FOR MODEL - LOG TO FILE

<font size="3">**04_get_metrics_function.py**</font> gets metrics as well as HOLDOUT\_ ACCURACY for a given model_id dereived from <font size="3">**db/pycaret_results.csv**</font>. It will diaply CM etc and then logs TEST_ID, model_id, accuracy to <font size="3">**csvs/test_results.csv**</font>.

By enabling rnd_num_split and rnd_num_hyper to be passed in, we can examine the effect of these random numbers.

## STEP 5: CREATE GENE IMPORTANCES

For a given model, <font size="3">**05_create_gene_importance_ensemble_log.py**</font> it runs model with a RND_STATE_HYPER, logs the RND_STATE_HYPER used in <font size="3">**csvs/gene_importances/rnd_used.csv**</font>. This prevents the double recording of gene importances, although witha range of 1 to 4_000_000_000 by default the likelihood is minimal but would be important for smaller random number range.

It then logs the gene importances based on model and code. AdaBoost has 100(give or take) non-zero values, so cut off point is straightfowrad. It will be different for RandFor, LogReg etc.

## STEP 6: GENE OCCURRENCE TOTALS

<font size="3">**0count_genes.py**</font> gets the total number of random numbers used from <font size="3">**csvs/gene_importances/rnd_used.csv**</font> and the value counts of each gene from <font size="3">**csvs/gene_importances/genes.csv**</font>.

TODO?

- add \_00X to end of rund_used and genes so that we have a job reference and can hae multiple jobs
- calculate holdout accuracy for each run and log that to a rnd_accuracy file or include in rnd_used...
