'''
Vedika Ahuja, Bhargavi Ganesh, and Pete Rodrigue
Script which creates feature importances and precision/recall curve for best model.
This script also creates a decision tree stump for the time split of the best model.
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
import graphviz
################################################################################
                        # SET GLOBALS FOR DATA CLEANING
################################################################################
offender_filepath = 'ncdoc_data/data/preprocessed/OFNT3CE1.csv'
inmate_filepath = 'ncdoc_data/data/preprocessed/INMT4BB1.csv'
demographics_filepath = 'ncdoc_data/data/preprocessed/OFNT3AA1.csv'
begin_date = '2007-01-01'
end_date = '2018-01-01'
recidivate_definition_in_days = 365
################################################################################
                        # SET GLOBALS FOR BEST MODEL
################################################################################
train_start = '2011-01-01'
train_end = '2011-12-31'
test_start = '2013-01-01'
test_end = '2013-12-31'
best_model_params = ensemble.RandomForestClassifier(max_depth=10, max_features='sqrt', n_estimators=1000)
################################################################################
OFNT3CE1 = pl.clean_offender_data(offender_filepath)
final_df = pl.make_final_df(offender_filepath, inmate_filepath, demographics_filepath, begin_date, end_date, recidivate_definition_in_days)
################################################################################
                        # TRAIN BEST MODEL
################################################################################
train_start_date = datetime.strptime(train_start, '%Y-%m-%d')
train_end_date = datetime.strptime(train_end, '%Y-%m-%d')
test_start_date = datetime.strptime(test_start, '%Y-%m-%d')
test_end_date = datetime.strptime(test_end, '%Y-%m-%d')

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
x_train, x_test, y_train, y_test = ml.temporal_split(final_df, time_var, pred_y, train_start, train_end, test_start, test_end, vars_to_drop_dates)
x_train, x_test, features = ml.pre_process(x_train, x_test, categorical_list, continuous_impute_list, vars_to_drop_all, source_of_count_vars, counts_vars)
x_train = x_train[features]
x_test = x_test[features]
y_pred_probs = best_model_params.fit(x_train, y_train).predict_proba(x_test)[:,1]
#Plot precision-recall
ml.plot_precision_recall_n(y_test, y_pred_probs, "Random Forest")
#Create feature importances
feature_importances = pd.DataFrame(best_model_params.feature_importances_, index = x_train.columns, columns=['importance']).sort_values('importance', ascending=False)
feature_importances.to_csv('output/feature_importance.csv')
#Print stump tree
dec_tree = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 1, min_samples_split= 2)
y_pred_probs_dec = dec_tree.fit(x_train, y_train).predict_proba(x_test)[:,1]
export_graphviz(
dec_tree, feature_names =x_train.columns, class_names=None, rounded=True, filled=True, out_file="output/plots/tree.dot")
graphviz.render('dot', 'png', "output/plots/tree.dot")
graphviz.Source.from_file("output/plots/tree.dot")
