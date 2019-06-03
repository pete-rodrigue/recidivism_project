#!/usr/bin/env python
# coding: utf-8
# Import packages
import pandas as pd
import numpy as np
import matplotlib as plt
import datetime
import pipeline_helper as ph
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import preprocess as pre
import pipeline_helper as ml
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

pd.options.display.max_columns = 100

################################################################################
                            # SET GLOBALS
################################################################################
offender_filepath = "ncdoc_data/data/preprocessed/OFNT3CE1.csv"
inmate_filepath = "ncdoc_data/data/preprocessed/INMT4BB1.csv"
demographics_filepath = "ncdoc_data/data/preprocessed/OFNT3AA1.csv"
begin_date = '2008-01-01'
end_date = '2018-01-01'
################################################################################
                        # SCRIPT - Merge and Format Data
################################################################################
OFNT3CE1 = clean_offender_data(offender_filepath)
INMT4BB1 = clean_inmate_data(inmate_filepath, begin_date, end_date)
merged = merge_offender_inmate_df(OFNT3CE1, INMT4BB1)
crime_w_release_date = collapse_counts_to_crimes(merged, begin_date)

#get rid of crimes outside of the whole period
df_to_ml_pipeline = crime_w_release_date.loc[crime_w_release_date['release_date_with_imputation'] > pd.to_datetime(begin_date)]
df_to_ml_pipeline = crime_w_release_date.loc[crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] < pd.to_datetime(end_date)]

#add recidivate label
crimes_w_time_since_release_date = create_time_since_last_release_df(crime_w_release_date)
crimes_w_recidviate_label = create_recidvate_label(crimes_w_time_since_release_date, 365)

#ask rayid about grace period
temp_split = bg_ml.temporal_dates(begin_date, end_date, prediction_windows, 0)

OFNT3AA1 = load_demographic_data(demographics_filepath)
crimes_w_demographic = crimes_w_recidviate_label.merge(OFNT3AA1,
                        on='OFFENDER_NC_DOC_ID_NUMBER',
                        how='left')

################################################################################
                # SCRIPT - Set pipeline parameters and train model
################################################################################

####################
#Pipeline parameters
####################

#Create temportal splits
prediction_windows = [12]
#note, we will have to start with the last end date possible before we collapse
#the counts by crime

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
    'RF':{'n_estimators': [10,100], 'max_depth': [5, 20, 100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,10,20,100],'min_samples_split': [2,5,10]},
    'SVM': {'C': [0.01]},
    'KNN': {'n_neighbors': [25],'weights': ['uniform','distance'],'algorithm': ['ball_tree']},
    'GB': {'n_estimators': [10], 'learning_rate': [0.1,0.5], 'subsample': [0.1,0.5], 'max_depth': [5]},
    'BG': {'n_estimators': [10], 'max_samples': [.5]}}

crimes_w_demographic.columns

# Index(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX',
#        'SENTENCE_EFFECTIVE(BEGIN)_DATE', 'release_date_with_imputation',
#        'crime_felony_or_misd', 'time_of_last_felony_release', 'recidivate',
#        'OFFENDER_BIRTH_DATE', 'OFFENDER_GENDER_CODE', 'OFFENDER_RACE_CODE',
#        'OFFENDER_HEIGHT_(IN_INCHES)', 'OFFENDER_WEIGHT_(IN_LBS)',
#        'OFFENDER_SKIN_COMPLEXION_CODE', 'OFFENDER_HAIR_COLOR_CODE',
#        'OFFENDER_EYE_COLOR_CODE', 'OFFENDER_BODY_BUILD_CODE',
#        'CITY_WHERE_OFFENDER_BORN', 'NC_COUNTY_WHERE_OFFENDER_BORN',
#        'STATE_WHERE_OFFENDER_BORN', 'COUNTRY_WHERE_OFFENDER_BORN',
#        'OFFENDER_CITIZENSHIP_CODE', 'OFFENDER_ETHNIC_CODE',
#        'OFFENDER_PRIMARY_LANGUAGE_CODE'],
#       dtype='object')
crimes_w_demographic.dtypes
pred_y = 'recidivate'
time_var = 'release_date_with_imputation'
to_dummy_list = []
vars_to_drop_all = ['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX', 'time_of_last_felony_release']
#Note - Make these into integer month and year variables
vars_to_drop_dates = ['release_date_with_imputation', 'SENTENCE_EFFECTIVE(BEGIN)_DATE', 'OFFENDER_BIRTH_DATE']
continuous_impute_list = []
categorical_list = ['crime_felony_or_misd', 'OFFENDER_GENDER_CODE', 'OFFENDER_RACE_CODE',
       'OFFENDER_HEIGHT_(IN_INCHES)', 'OFFENDER_WEIGHT_(IN_LBS)',
       'OFFENDER_SKIN_COMPLEXION_CODE', 'OFFENDER_HAIR_COLOR_CODE',
       'OFFENDER_EYE_COLOR_CODE', 'OFFENDER_BODY_BUILD_CODE',
       'CITY_WHERE_OFFENDER_BORN', 'NC_COUNTY_WHERE_OFFENDER_BORN',
       'STATE_WHERE_OFFENDER_BORN', 'COUNTRY_WHERE_OFFENDER_BORN',
       'OFFENDER_CITIZENSHIP_CODE', 'OFFENDER_ETHNIC_CODE',
       'OFFENDER_PRIMARY_LANGUAGE_CODE']
outfile = 'output/test_pipeline.csv'
temp_split_sub = temp_split[0]
temp_split_sub

crimes_w_demographic.columns
crimes_w_demographic.isnull().any()
temp_split_sub[0]
x_train, x_test, y_train, y_test = bg_ml.temporal_split(crimes_w_demographic, time_var, pred_y, temp_split_sub[0], temp_split_sub[1], temp_split_sub[2], temp_split_sub[3], vars_to_drop_dates)
x_train, x_test, features = bg_ml.pre_process(x_train, x_test, categorical_list, to_dummy_list, continuous_impute_list, vars_to_drop_all)
#build models
# x_train.select_dtypes(include=[np.datetime64])
x_train = x_train[features]
x_test = x_test[features]
y_pred_probs = tree.DecisionTreeClassifier().fit(x_train, y_train).predict_proba(x_test)[:,1]
results_df, params = run_models(models_to_run, classifiers, parameters, crimes_w_demographic, pred_y, temp_split_sub, time_var, categorical_list, to_dummy_list, continuous_impute_list, vars_to_drop_all, vars_to_drop_dates, k_list, outfile)















# grid_size = 'test'
# temporal_split_date_var = 'release_date_with_imputation'
# testing_length = 12
# grace_period = 0
# validation_dates = ['2009-01-01']
# # models_to_run = ['DT', 'RF', 'LR', 'GB', 'AB']
# # did not run these models - 'SVM', 'KNN'
# models_to_run = ['DT']
#
# #inputs into the preprocess function - need to tell the function which variables to clean
# columns_to_datetime = ['time_of_last_felony_release', 'release_date_with_imputation', 'SENTENCE_EFFECTIVE(BEGIN)_DATE', 'OFFENDER_BIRTH_DATE']
# dummy_vars = ['crime_felony_or_misd', 'OFFENDER_GENDER_CODE', 'OFFENDER_RACE_CODE', 'STATE_WHERE_OFFENDER_BORN', 'NC_COUNTY_WHERE_OFFENDER_BORN', 'CITY_WHERE_OFFENDER_BORN', 'NC_COUNTY_WHERE_OFFENDER_BORN',
#                     'STATE_WHERE_OFFENDER_BORN', 'COUNTRY_WHERE_OFFENDER_BORN', 'OFFENDER_CITIZENSHIP_CODE', 'OFFENDER_ETHNIC_CODE',
#                     'OFFENDER_PRIMARY_LANGUAGE_CODE']
#
# boolean_vars = []
# #some variables are id vars that we don't want to include these as features
# vars_not_to_include = ['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX']
# prediction_var = 'recidivate'
#
# for validation_date in validation_dates:
#     train_set, validation_set = ml.temporal_split(crimes_w_demographic, temporal_split_date_var, validation_date, testing_length, grace_period)
#
#     #preprocess the train_set and test_set separately
#     train_set = pre.pre_process(train_set, dummy_vars, boolean_vars, vars_not_to_include, columns_to_datetime)
#     validation_set = pre.pre_process(validation_set, dummy_vars, boolean_vars, vars_not_to_include, columns_to_datetime)
#
#     #create features - there will be features in the train that don't exist in test and vice versa
#     #the model will only actually use the union of the two.
#     train_features  = list(train_set.columns)
#     test_features = list(validation_set.columns)
#
#     #find union of the two lists
#     intersection_features = list(set(train_features) & set(test_features))
#     intersection_features.remove(prediction_var)
#
#     #run the loop and save the output df
#     results_df = ml.clf_loop(train_set, validation_set, intersection_features, prediction_var, models_to_run, clfs, grid, results_df, validation_date, outfile)


#add features and run!!!
                    # READ AND CLEAN OFNT3CE1
################################################################################
def clean_offender_data(offender_filepath):
    '''
    Takes the offender dataset (OFNT3CE1), cleans it, and outputs it as a to_csv
    '''
    OFNT3CE1 = pd.read_csv(offender_filepath,
        dtype={'OFFENDER_NC_DOC_ID_NUMBER': str,
               'MAXIMUM_SENTENCE_LENGTH': str,
               'SPLIT_SENTENCE_ACTIVE_TERM': str,
               'SENTENCE_TYPE_CODE.5': str,
               'PRIOR_P&P_COMMNT/COMPONENT_ID': str,
               'ORIGINAL_SENTENCE_AUDIT_CODE': str})

    OFNT3CE1.shape  # Number of rows
    pd.options.display.max_columns = 100  # Set the max number of col to display

    # Only keep people that have ever had a felony offense
    # Create a variable that indicates felony offenses
    OFNT3CE1['has_felony'] = np.where(
        OFNT3CE1['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0).copy()
    OFNT3CE1 = OFNT3CE1.groupby(
        'OFFENDER_NC_DOC_ID_NUMBER').filter(lambda x: x['has_felony'].max() == 1)
    OFNT3CE1.shape  # Notice we have fewer rows now

    #clean the dates
    OFNT3CE1['clean_SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(
            OFNT3CE1['SENTENCE_EFFECTIVE(BEGIN)_DATE'], errors='coerce')


    # dropping features we don't want to use:
    OFNT3CE1 = OFNT3CE1.drop(['NC_GENERAL_STATUTE_NUMBER',
                              'LENGTH_OF_SUPERVISION',
                              'SUPERVISION_TERM_EXTENSION',
                              'SUPERVISION_TO_FOLLOW_INCAR.',
                              'G.S._MAXIMUM_SENTENCE_ALLOWED',
                              'ICC_JAIL_CREDITS_(IN_DAYS)'], axis=1)

    # Making one person's id a number so we can make them all numeric
    OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].loc[
            OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == 'T153879'] = "-999"
    OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] = pd.to_numeric(
            OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'])

    # OFNT3CE1.to_csv("OFNT3CE1.csv")
    return OFNT3CE1


################################################################################
                        # CLEAN AND READ  INMT4BB1
################################################################################
def clean_inmate_data(inmate_filepath, begin_date, end_date):
    '''
    Reads and cleans the inmate data.

    Inputs:
        - inmate_filepath: csv file path
        - begin_date: The beginning date of the time period of interest -
                        This is the release date we want to look at
        - end_date: The end date of the time period
    '''
    INMT4BB1 = pd.read_csv(inmate_filepath)
    INMT4BB1.head()

    # dropping features we don't want to use:
    INMT4BB1 = INMT4BB1.drop(['INMATE_COMPUTATION_STATUS_FLAG',
                              'PAROLE_DISCHARGE_DATE',
                              'PAROLE_SUPERVISION_BEGIN_DATE'], axis=1)
    #clean dates
    INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'] = pd.to_datetime(
                INMT4BB1['ACTUAL_SENTENCE_END_DATE'], errors='coerce')
    INMT4BB1['clean_projected_release_date'] = pd.to_datetime(
                INMT4BB1['PROJECTED_RELEASE_DATE_(PRD)'], errors='coerce')
    INMT4BB1['clean_SENTENCE_BEGIN_DATE_(FOR_MAX)'] = pd.to_datetime(
            INMT4BB1['SENTENCE_BEGIN_DATE_(FOR_MAX)'], errors='coerce')

    INMT4BB1['release_date_with_imputation'] = INMT4BB1[
                                'clean_ACTUAL_SENTENCE_END_DATE']
    INMT4BB1.head()
    INMT4BB1['release_date_with_imputation'] = np.where(
        (INMT4BB1['release_date_with_imputation'].isnull()),
        INMT4BB1['clean_projected_release_date'],
        INMT4BB1['release_date_with_imputation']).copy()

    INMT4BB1['imputed_release_date_flag'] = np.where(
            INMT4BB1['clean_projected_release_date'].notnull() &
            INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'].isnull(), 1, 0).copy()

    INMT4BB1.tail(10)

    # Number of remaining people
    INMT4BB1['INMATE_DOC_NUMBER'].unique().shape

    return INMT4BB1


def load_demographic_data(demographics_filepath):
    '''Loads and cleans the demographic dataset'''
    OFNT3AA1 = pd.read_csv(demographics_filepath, dtype={
        'OFFENDER_BIRTH_DATE': str,
        'OFFENDER_GENDER_CODE': str,
        'OFFENDER_RACE_CODE': str,
        'OFFENDER_SKIN_COMPLEXION_CODE': str,
        'OFFENDER_HAIR_COLOR_CODE': str,
        'OFFENDER_EYE_COLOR_CODE': str,
        'OFFENDER_BODY_BUILD_CODE': str,
        'CITY_WHERE_OFFENDER_BORN': str,
        'NC_COUNTY_WHERE_OFFENDER_BORN': str,
        'STATE_WHERE_OFFENDER_BORN': str,
        'COUNTRY_WHERE_OFFENDER_BORN': str,
        'OFFENDER_CITIZENSHIP_CODE': str,
        'OFFENDER_ETHNIC_CODE': str,
        'OFFENDER_PRIMARY_LANGUAGE_CODE': str})
    OFNT3AA1 = OFNT3AA1.drop(['OFFENDER_SHIRT_SIZE', 'OFFENDER_PANTS_SIZE',
                   'OFFENDER_JACKET_SIZE', 'OFFENDER_SHOE_SIZE',
                   'OFFENDER_DRESS_SIZE', 'NEXT_PHOTO_YEAR',
                   'DATE_OF_LAST_UPDATE', 'TIME_OF_LAST_UPDATE'], axis=1)

    OFNT3AA1['OFFENDER_HEIGHT_(IN_INCHES)'] = pd.to_numeric(
            OFNT3AA1['OFFENDER_HEIGHT_(IN_INCHES)'])
    OFNT3AA1['OFFENDER_WEIGHT_(IN_LBS)'] = pd.to_numeric(
            OFNT3AA1['OFFENDER_WEIGHT_(IN_LBS)'])

    OFNT3AA1['OFFENDER_NC_DOC_ID_NUMBER'] = OFNT3AA1[
                'OFFENDER_NC_DOC_ID_NUMBER'].astype(str)
    OFNT3AA1['OFFENDER_NC_DOC_ID_NUMBER'] = OFNT3AA1[
                    'OFFENDER_NC_DOC_ID_NUMBER'].str.replace(
                    'T', '', regex=False)

    OFNT3AA1['OFFENDER_NC_DOC_ID_NUMBER'] = pd.to_numeric(
            OFNT3AA1['OFFENDER_NC_DOC_ID_NUMBER'])

    return OFNT3AA1

def merge_offender_inmate_df(OFNT3CE1, INMT4BB1):
    '''
    Merge the inmate and offender pandas dataframes.

    Inputs:
        - OFNT3CE1: offender pandas dataframes
        - INMT4BB1: inmates pandas dataframe
    '''
    # OFNT3CE1.dtypes
    merged = OFNT3CE1.merge(INMT4BB1,
                            left_on=['OFFENDER_NC_DOC_ID_NUMBER',
                                     'COMMITMENT_PREFIX',
                                     'SENTENCE_COMPONENT_NUMBER'],
                            right_on=['INMATE_DOC_NUMBER',
                                      'INMATE_COMMITMENT_PREFIX',
                                      'INMATE_SENTENCE_COMPONENT'],
                            how='outer')
    merged.head()
    # Find the people who have been to jail that we want to keep
    ids_to_keep = merged['INMATE_DOC_NUMBER'].unique()
    ids_to_keep = ids_to_keep[~np.isnan(ids_to_keep)]
    merged = merged.loc[merged['OFFENDER_NC_DOC_ID_NUMBER'].isin(
                list(ids_to_keep)), : ]

    return merged


def collapse_counts_to_crimes(merged, begin_date):
    '''
    Create a dataframe to put into the ml pipeline.

    The dataframe to put in the pipeline will have 1 row for every crime
    (instead of for every count)

    Inputs:
        - merged: the merged offender and inmate dataframe
        - begin date
        - end date
    '''
    #filter for the counts with release dates before the timeframe
    #we'll filter for crimes with release dates after the timeframe
    time_mask = (merged['release_date_with_imputation'] > begin_date)
    final = merged[time_mask]
    final['release_date_with_imputation'].describe()

    #collapse all counts of a crime into one event
    final['crime_felony_or_misd'] = np.where(final['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0).copy()
    crime_label = final.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX']).apply(lambda x: x['crime_felony_or_misd'].sum()).to_frame().reset_index(
                        ).rename(columns={0: 'num_of_felonies'})

    crime_label['crime_felony_or_misd'] = np.where(crime_label['num_of_felonies'] > 0, 'FELON', 'MISD').copy()

    #assign a begin date and an end date to each crime
    release_date = final.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX']
                                    ).agg({'release_date_with_imputation': 'max',
                                           'SENTENCE_EFFECTIVE(BEGIN)_DATE': 'min',
                                           'SENTENCE_COMPONENT_NUMBER' : 'count'}
                                    ).reset_index().rename(columns={'SENTENCE_COMPONENT_NUMBER': 'counts_per_crime'})

    #merge together to know if a crime is a misdeamonor or felony
    crime_w_release_date = release_date.merge(crime_label, on=['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'], how='outer')

    crime_w_release_date = crime_w_release_date.sort_values(['OFFENDER_NC_DOC_ID_NUMBER', 'release_date_with_imputation'])
    crime_w_release_date = crime_w_release_date[['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX', 'SENTENCE_EFFECTIVE(BEGIN)_DATE', 'release_date_with_imputation', 'crime_felony_or_misd']]
    crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'])
    return crime_w_release_date


def create_time_since_last_release_df(crime_df):
    '''
    Creates a dataframe unique on OFFENDER_NC_DOC_ID_NUMBER and COMMITMENT_PREFIX (a person and a crime),
    and indicates the time since the person's last felony.

    Helper function for create_df_for_ml_pipeline
    '''
    for index in range(1, crime_df.shape[0]):
      for reverse_index in range(index-1, -1, -1):
        # if the past row is the same person id:
        if crime_df.loc[index, 'OFFENDER_NC_DOC_ID_NUMBER'] == crime_df.loc[reverse_index, 'OFFENDER_NC_DOC_ID_NUMBER']:
          if crime_df.loc[reverse_index, 'crime_felony_or_misd'] == 'FELON':
            crime_df.loc[index, 'time_of_last_felony_release'] = crime_df.loc[reverse_index, 'release_date_with_imputation']
            break
        # if the past row is NOT the same person id, go to the next row
        else:
          break

    return crime_df

def create_recidvate_label(crime_w_release_date, recidviate_definition_in_days):
    crime_w_release_date['recidivate'] = 0
    diff = crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] - crime_w_release_date['time_of_last_felony_release']
    crime_w_release_date.loc[diff < pd.to_timedelta(365, 'D'), 'recidivate'] = 1

    return crime_w_release_date


################################################################################
                        # ADD FEATURES
                        #rough work
################################################################################

# ADD FEATURES
# create a new data frame that has the total number of incidents with the law
# for each person (guilty or innocent, went to jail or not)
total_number_of_counts = merged.groupby('OFFENDER_NC_DOC_ID_NUMBER').count(
        )['COMMITMENT_PREFIX'].to_frame().reset_index(
        ).rename(columns={'COMMITMENT_PREFIX': 'total_num_counts'})
total_number_of_counts.head()

# create a new data frame that has the total number of incarceral events
total_number_of_incarcerations = merged.groupby(
    ['INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX']).count(
    ).reset_index(
    )[['INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX']].groupby(
            'INMATE_DOC_NUMBER').count().reset_index().rename(
            columns={
            'INMATE_COMMITMENT_PREFIX': 'total_number_of_incarcerations'})

total_number_of_incarcerations.head()
total_number_of_incarcerations.describe()
myhist = total_number_of_incarcerations['total_number_of_incarcerations'].hist()
total_number_of_incarcerations['total_number_of_incarcerations'].quantile(.99)
merged.head()
# did they recidivate within 2 years of last arrest?
#flag earliest time released in the given time period
#time between each release and the release before it
#if there is only one release, then recidivate dummy = 0
#if there is more than one release, and less than 24 months earlier, then recidivate
#dummy is 1.

merged.groupby('INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX')

'release_date_with_imputation'













































first_release = merged.sort_values(
    "clean_ACTUAL_SENTENCE_END_DATE").groupby(
    "OFFENDER_NC_DOC_ID_NUMBER", as_index=False)[
    'clean_ACTUAL_SENTENCE_END_DATE'].first()

merged = merged.merge(first_release,
                      on='OFFENDER_NC_DOC_ID_NUMBER',
                      how='left')
merged.head()

merged = merged.rename(columns={"clean_ACTUAL_SENTENCE_END_DATE_x":
                                "clean_ACTUAL_SENTENCE_END_DATE",
                                "clean_ACTUAL_SENTENCE_END_DATE_y":
                                "first_release_date"})



# Should we use SENTENCE_BEGIN_DATE_(FOR_MAX) or SENTENCE_EFFECTIVE(BEGIN)_DATE
# here?
merged.shape
merged.loc[merged[
        'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
        merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], :].shape
merged.loc[merged[
        'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
        merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], : ][['OFFENDER_NC_DOC_ID_NUMBER','SENTENCE_BEGIN_DATE_(FOR_MAX)', 'SENTENCE_EFFECTIVE(BEGIN)_DATE']]









merged.loc[merged['first_release_date'].isnull()==False, :].head(10)
merged.loc[merged['first_release_date'].isnull()==False, :].groupby('OFFENDER_NC_DOC_ID_NUMBER').count()

merged['time_elapsed'] = merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'] - \
                         merged['first_release_date']
merged['outcome'] = 0
merged.loc[(merged['time_elapsed'] >= '0 days') &
           (merged['time_elapsed'] <= '730 days'), 'outcome'] = 1

merged.loc[merged['first_release_date'].isnull()==False, :].head(10)
merged.head(11)

# Do we want to only keep people with real values for
# clean_ACTUAL_SENTENCE_END_DATE?


# Watch out for cases like this where we have consecutive sentences
merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114, :]

merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114,
                      ['DATE_OFFENSE_COMMITTED_-_BEGIN',
                      'DATE_OFFENSE_COMMITTED_-_END',
                      'OFFENDER_NC_DOC_ID_NUMBER',
                      'SENTENCE_BEGIN_DATE_(FOR_MAX)',
                      'clean_ACTUAL_SENTENCE_END_DATE',
                      'first_release_date',
                      'time_elapsed',
                      'outcome']]


INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 62, :]
OFNT3CE1.loc[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == 62, :]
merged.loc[merged['OFFENDER_NC_DOC_ID_NUMBER'] == 62,
                      ['PROJECTED_RELEASE_DATE_(PRD)',
                      'DATE_OFFENSE_COMMITTED_-_BEGIN',
                      'DATE_OFFENSE_COMMITTED_-_END',
                      'OFFENDER_NC_DOC_ID_NUMBER',
                      'SENTENCE_BEGIN_DATE_(FOR_MAX)',
                      'clean_ACTUAL_SENTENCE_END_DATE',
                      'first_release_date',
                      'time_elapsed',
                      'outcome']]


# NOTE TODO: aggregate variables and collapse to the most recent incarceration
# Eventually we will only want to keep the row with a
# clean_ACTUAL_SENTENCE_END_DATE that matches first_release_date
# That will require spreading outcome within each person.
merged.groupby('OFFENDER_NC_DOC_ID_NUMBER')['outcome'].max()


















OFNT9BE1 = pd.read_csv(
    "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\ncdoc_data\\data\\preprocessed\\OFNT9BE1.csv")
    crime_w_release_date = release_date.merge(crime_label,
                            on=['OFFENDER_NC_DOC_ID_NUMBER',
                                     'COMMITMENT_PREFIX'],
                            how='outer')
    # crime_w_release_date = crime_w_release_date.drop(columns=['num_of_felonies'], axis=1)
    crime_w_release_date.head(5)
    #create a variable of time since last release
    crime_w_release_date['release_date_final'] = pd.to_datetime(crime_w_release_date['crime_release_data_w_imputation'])
    diff_in_dates = np.where(crime_w_release_date['num_of_felonies'] > 1, (crime_w_release_date.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'])['release_date_final'].diff(periods=-1)), 0)
    diff_in_dates
    test = crime_w_release_date.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'])['release_date_final'].diff().reset_index()
    test
    # test.describe()
    # time_since_last_release =
#
#
#
#
# #     diff_in_dates = crime_w_release_date.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'])[].diff(periods=-1, axis=0)
# #     diff_in_dates
# #
# #     #drop
# #
# # crime.head()
# # ADD FEATURES
# # create a new data frame that has the total number of incidents with the law
# # for each person (guilty or innocent, went to jail or not)
# total_number_of_counts = merged.groupby('OFFENDER_NC_DOC_ID_NUMBER').count(
#         )['COMMITMENT_PREFIX'].to_frame().reset_index(
#         ).rename(columns={'COMMITMENT_PREFIX': 'total_num_counts'})
# total_number_of_counts.head()
#
# # create a new data frame that has the total number of incarceral events
# total_number_of_incarcerations = merged.groupby(
#     ['INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX']).count(
#     ).reset_index(
#     )[['INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX']].groupby(
#             'INMATE_DOC_NUMBER').count().reset_index().rename(
#             columns={
#             'INMATE_COMMITMENT_PREFIX': 'total_number_of_incarcerations'})
#
# total_number_of_incarcerations.head()
# total_number_of_incarcerations.describe()
# myhist = total_number_of_incarcerations['total_number_of_incarcerations'].hist()
# total_number_of_incarcerations['total_number_of_incarcerations'].quantile(.99)
# merged.head()
# # did they recidivate within 2 years of last arrest?
# #flag earliest time released in the given time period
# #time between each release and the release before it
# #if there is only one release, then recidivate dummy = 0
# #if there is more than one release, and less than 24 months earlier, then recidivate
# #dummy is 1.
#
# merged.groupby('INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX')
#
# 'release_date_with_imputation'
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# first_release = merged.sort_values(
#     "clean_ACTUAL_SENTENCE_END_DATE").groupby(
#     "OFFENDER_NC_DOC_ID_NUMBER", as_index=False)[
#     'clean_ACTUAL_SENTENCE_END_DATE'].first()
#
# merged = merged.merge(first_release,
#                       on='OFFENDER_NC_DOC_ID_NUMBER',
#                       how='left')
# merged.head()
#
# merged = merged.rename(columns={"clean_ACTUAL_SENTENCE_END_DATE_x":
#                                 "clean_ACTUAL_SENTENCE_END_DATE",
#                                 "clean_ACTUAL_SENTENCE_END_DATE_y":
#                                 "first_release_date"})
#
#
#
# # Should we use SENTENCE_BEGIN_DATE_(FOR_MAX) or SENTENCE_EFFECTIVE(BEGIN)_DATE
# # here?
# merged.shape
# merged.loc[merged[
#         'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
#         merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], :].shape
# merged.loc[merged[
#         'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
#         merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], : ][['OFFENDER_NC_DOC_ID_NUMBER','SENTENCE_BEGIN_DATE_(FOR_MAX)', 'SENTENCE_EFFECTIVE(BEGIN)_DATE']]
#
#
#
#
#
#
#
#
#
# merged.loc[merged['first_release_date'].isnull()==False, :].head(10)
# merged.loc[merged['first_release_date'].isnull()==False, :].groupby('OFFENDER_NC_DOC_ID_NUMBER').count()
#
# merged['time_elapsed'] = merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'] - \
#                          merged['first_release_date']
# merged['outcome'] = 0
# merged.loc[(merged['time_elapsed'] >= '0 days') &
#            (merged['time_elapsed'] <= '730 days'), 'outcome'] = 1
#
# merged.loc[merged['first_release_date'].isnull()==False, :].head(10)
# merged.head(11)
#
# # Do we want to only keep people with real values for
# # clean_ACTUAL_SENTENCE_END_DATE?
#
#
# # Watch out for cases like this where we have consecutive sentences
# merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114, :]
#
# merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114,
#                       ['DATE_OFFENSE_COMMITTED_-_BEGIN',
#                       'DATE_OFFENSE_COMMITTED_-_END',
#                       'OFFENDER_NC_DOC_ID_NUMBER',
#                       'SENTENCE_BEGIN_DATE_(FOR_MAX)',
#                       'clean_ACTUAL_SENTENCE_END_DATE',
#                       'first_release_date',
#                       'time_elapsed',
#                       'outcome']]
#
#
# INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 62, :]
# OFNT3CE1.loc[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == 62, :]
# merged.loc[merged['OFFENDER_NC_DOC_ID_NUMBER'] == 62,
#                       ['PROJECTED_RELEASE_DATE_(PRD)',
#                       'DATE_OFFENSE_COMMITTED_-_BEGIN',
#                       'DATE_OFFENSE_COMMITTED_-_END',
#                       'OFFENDER_NC_DOC_ID_NUMBER',
#                       'SENTENCE_BEGIN_DATE_(FOR_MAX)',
#                       'clean_ACTUAL_SENTENCE_END_DATE',
#                       'first_release_date',
#                       'time_elapsed',
#                       'outcome']]
#
#
# # NOTE TODO: aggregate variables and collapse to the most recent incarceration
# # Eventually we will only want to keep the row with a
# # clean_ACTUAL_SENTENCE_END_DATE that matches first_release_date
# # That will require spreading outcome within each person.
# merged.groupby('OFFENDER_NC_DOC_ID_NUMBER')['outcome'].max()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# OFNT9BE1 = pd.read_csv(
#     "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\ncdoc_data\\data\\preprocessed\\OFNT9BE1.csv")
