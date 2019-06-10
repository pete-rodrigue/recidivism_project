#!/usr/bin/env python
# coding: utf-8
# Import packages
import pandas as pd
import numpy as np
import matplotlib as plt
import datetime
# import pipeline_helper as ph
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
# import preprocess as pre
# import pipeline_helper as ml
import assignment3_functions_bg as bg_ml
from sklearn import (svm, ensemble, tree,
                     linear_model, neighbors, naive_bayes, dummy)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_curve
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import ParameterGrid

print('\tmodules loaded, now loading data\t', datetime.now())
pd.options.display.max_columns = 100
################################################################################
                            # SET GLOBALS
################################################################################
begin_date = '2007-01-01'
end_date = '2018-01-01'




final_df = pd.read_csv('C:\\Users\\edwar.WJM-SONYLAPTOP\\Documents\GitHub\\recidivism_project\\final_final_df.csv')
print('\tdata loaded\t', datetime.now())
final_df.dtypes
final_df['release_date_with_imputation'] = pd.to_datetime(
        final_df['release_date_with_imputation'])
final_df['release_date_with_imputation'].head()

################################################################################
                # SCRIPT - Set pipeline parameters and train model
################################################################################

####################
#Pipeline parameters
####################

#Create temporal splits
# prediction_windows = [12]
# temp_split = bg_ml.temporal_dates(begin_date, end_date, prediction_windows, 0)
#note, we will have to start with the last end date possible before we collapse
#the counts by crime
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

temp_split
## ML Pipeline parameters
models_to_run = ['DT']
k_list = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]

classifiers = {'RF': ensemble.RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'LR': linear_model.LogisticRegression(penalty='l1', C=1e5, n_jobs=-1),
    'SVM': svm.LinearSVC(tol= 1e-5, random_state=0),
    'AB': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'DT': tree.DecisionTreeClassifier(),
    'KNN': neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    'GB': ensemble.GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
    'BG': ensemble.BaggingClassifier(linear_model.LogisticRegression(penalty='l1', C=1e5, n_jobs=-1))
        }

parameters = {
    'RF':{'n_estimators': [10, 100], 'max_depth': [10, 20, 50], 'max_features': ['sqrt','log2'], 'min_samples_split': [10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'min_samples_split': [2]},
    'SVM': {'C': [0.01]},
    'KNN': {'n_neighbors': [15, 25],'weights': ['uniform','distance'],'algorithm': ['ball_tree']},
    'GB': {'n_estimators': [10], 'learning_rate': [0.1,0.5], 'subsample': [0.1,0.5], 'max_depth': [5]},
    'BG': {'n_estimators': [10], 'max_samples': [.5]}}

test = {'DT': {'criterion': ['gini'], 'max_depth': [1], 'min_samples_split': [2]}}

pred_y = 'recidivate'
time_var = 'release_date_with_imputation'
to_dummy_list = []
vars_to_drop_all = ['index', 'OFFENDER_NC_DOC_ID_NUMBER',
                    'COMMITMENT_PREFIX',
                    'start_time_of_next_incarceration',
                    'crime_felony_or_misd']
#Note - Make these into integer month and year variables
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

outfile = 'test_pipeline.csv'
temp_split_sub = temp_split[0]

print("\ttraining models\t", datetime.now())
results_df, params = bg_ml.run_models(models_to_run,
                                      classifiers,
                                      parameters,
                                      final_df,
                                      pred_y, temp_split,
                                      time_var,
                                      categorical_list,
                                      to_dummy_list,
                                      continuous_impute_list,
                                      vars_to_drop_all,
                                      vars_to_drop_dates,
                                      k_list,
                                      outfile)
