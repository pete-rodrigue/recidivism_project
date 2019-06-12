'''
Vedika Ahuja, Bhargavi Ganesh, and Pete Rodrigue

Script to run through pipeline. Specifies parameters, and uses functions in
ml_functions and processing_library files to create a table of evaluation metrics
comparing different models across different parameters.
'''
import pandas as pd
import numpy as np
import matplotlib as plt
import datetime
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from sklearn import (svm, ensemble, tree,
                     linear_model, neighbors, naive_bayes, dummy)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_curve
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import ParameterGrid
import processing_library as pl
import ml_functions_library as ml
import data_explore_library as de

################################################################################
                            # SET GLOBALS
################################################################################
offender_filepath = 'ncdoc_data/data/preprocessed/OFNT3CE1.csv'
inmate_filepath = 'ncdoc_data/data/preprocessed/INMT4BB1.csv'
demographics_filepath = 'ncdoc_data/data/preprocessed/OFNT3AA1.csv'
begin_date = '2007-01-01'
end_date = '2018-01-01'
recidivate_definition_in_days = 365
################################################################################
                    # SCRIPT - Load and Format Data
################################################################################
#Clean different tables and merge
OFNT3CE1 = pl.clean_offender_data(offender_filepath)
final_df = pl.make_final_df(offender_filepath, inmate_filepath, demographics_filepath, begin_date, end_date, recidivate_definition_in_days)
################################################################################
                # SCRIPT - Set pipeline parameters and train model
################################################################################
####################
#Pipeline parameters
####################
temp_split = [[datetime.strptime('2007-01-01', '%Y-%m-%d'),
              datetime.strptime('2007-12-31', '%Y-%m-%d'),
              datetime.strptime('2009-01-01', '%Y-%m-%d'),
              datetime.strptime('2009-12-31', '%Y-%m-%d')],
              [datetime.strptime('2011-01-01', '%Y-%m-%d'),
               datetime.strptime('2011-12-31', '%Y-%m-%d'),
               datetime.strptime('2013-01-01', '%Y-%m-%d'),
               datetime.strptime('2013-12-31', '%Y-%m-%d')],
              [datetime.strptime('2015-01-01', '%Y-%m-%d'),
               datetime.strptime('2015-12-31', '%Y-%m-%d'),
               datetime.strptime('2017-01-01', '%Y-%m-%d'),
               datetime.strptime('2017-12-31', '%Y-%m-%d')]]

## ML Pipeline parameters
k_list = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]
classifiers = {'RF': ensemble.RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'LR': linear_model.LogisticRegression(penalty='l1', C=1e5, n_jobs=-1, solver='liblinear'),
    'SVM': svm.LinearSVC(tol= 1e-5, random_state=0),
    'AB': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'DT': tree.DecisionTreeClassifier(),
    'KNN': neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    'GB': ensemble.GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
    'BG': ensemble.BaggingClassifier(tree.DecisionTreeClassifier())
        }
models_to_run = ['RF', 'LR', 'AB', 'DT', 'SVM', 'KNN', 'GB', 'BG']
parameters = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [10, 100, 500], 'max_features': ['sqrt'], 'min_samples_split': [2, 10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1,10,100,1000,10000]},
    'DT': {'criterion': ['gini'], 'max_depth': [1, 10, 20, 100, None], 'min_samples_split': [2, 5, 10]},
    'SVM': {'C': [0.01]},
    'KNN': {'n_neighbors': [15, 25],'weights': ['uniform','distance'],'algorithm': ['ball_tree']},
    'GB': {'n_estimators': [10, 50, 100], 'learning_rate': [0.1,0.5], 'subsample': [0.1,0.5], 'max_depth': [5, 10]},
    'BG': {'n_estimators': [10, 50, 100], 'max_samples': [.5]}}

pred_y = 'recidivate'
time_var = 'release_date_with_imputation'
vars_to_drop_all = ['index', 'OFFENDER_NC_DOC_ID_NUMBER',
                    'COMMITMENT_PREFIX',
                    'start_time_of_next_incarceration',
                    'crime_felony_or_misd']
vars_to_drop_dates = ['release_date_with_imputation',
                      'SENTENCE_EFFECTIVE(BEGIN)_DATE',
                      'OFFENDER_BIRTH_DATE']
continuous_impute_list = ['OFFENDER_HEIGHT_(IN_INCHES)', 'OFFENDER_WEIGHT_(IN_LBS)']
categorical_list = ['OFFENDER_GENDER_CODE', 'OFFENDER_RACE_CODE',
'OFFENDER_SKIN_COMPLEXION_CODE', 'OFFENDER_HAIR_COLOR_CODE',
'OFFENDER_EYE_COLOR_CODE', 'OFFENDER_BODY_BUILD_CODE',
'CITY_WHERE_OFFENDER_BORN', 'NC_COUNTY_WHERE_OFFENDER_BORN',
'STATE_WHERE_OFFENDER_BORN', 'COUNTRY_WHERE_OFFENDER_BORN',
'OFFENDER_CITIZENSHIP_CODE', 'OFFENDER_ETHNIC_CODE',
'OFFENDER_PRIMARY_LANGUAGE_CODE']
source_of_count_vars = OFNT3CE1
counts_vars = ['COUNTY_OF_CONVICTION_CODE',
                'PUNISHMENT_TYPE_CODE',
                'COMPONENT_DISPOSITION_CODE',
                'PRIMARY_OFFENSE_CODE',
                'COURT_TYPE_CODE',
                'SENTENCING_PENALTY_CLASS_CODE']
outfile = 'output/final_evaluation3.csv'
print("\ttraining models\t", datetime.now())
results_df, params = ml.run_models(models_to_run,
                                   classifiers,
                                   parameters,
                                   final_df,
                                   pred_y, temp_split,
                                   time_var,
                                   categorical_list,
                                   continuous_impute_list,
                                   vars_to_drop_all,
                                   vars_to_drop_dates,
                                   k_list,
                                   source_of_count_vars, counts_vars,
                                   outfile)
