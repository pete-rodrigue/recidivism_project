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

print('\tmodules loaded\t', datetime.now())
pd.options.display.max_columns = 100

                    # READ AND CLEAN OFNT3CE1
################################################################################
def clean_offender_data(offender_filepath):
    '''
    Takes the offender dataset (OFNT3CE1), cleans it, and outputs it as a to_csv
    '''
    # offender_filepath = 'C:/Users/edwar.WJM-SONYLAPTOP/Desktop/ncdoc_data/data/preprocessed/OFNT3CE1.csv'
    OFNT3CE1 = pd.read_csv(offender_filepath,
        dtype={'OFFENDER_NC_DOC_ID_NUMBER': str,
               'MAXIMUM_SENTENCE_LENGTH': str,
               'SPLIT_SENTENCE_ACTIVE_TERM': str,
               'SENTENCE_TYPE_CODE.5': str,
               'PRIOR_P&P_COMMNT/COMPONENT_ID': str,
               'ORIGINAL_SENTENCE_AUDIT_CODE': str})


    # Create a variable that indicates felony offenses
    OFNT3CE1['is_felony'] = np.where(
        OFNT3CE1['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0)
    doc_ids_with_felony = OFNT3CE1.groupby(
        'OFFENDER_NC_DOC_ID_NUMBER').filter(
                lambda x: x['is_felony'].max() == 1).reset_index(
                )['OFFENDER_NC_DOC_ID_NUMBER'].unique().tolist()
    OFNT3CE1 = OFNT3CE1[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].isin(doc_ids_with_felony)]
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
    INMT4BB1['release_date_with_imputation'] = np.where(
        (INMT4BB1['release_date_with_imputation'].isnull()),
        INMT4BB1['clean_projected_release_date'],
        INMT4BB1['release_date_with_imputation']).copy()

    INMT4BB1['imputed_release_date_flag'] = np.where(
            INMT4BB1['clean_projected_release_date'].notnull() &
            INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'].isnull(), 1, 0).copy()

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
                            how='right')

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
    # final['release_date_with_imputation'].describe()

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
                                    ).reset_index().rename(columns={'SENTENCE_COMPONENT_NUMBER': 'total_counts_for_crime'})

    #merge together to know if a crime is a misdeamonor or felony
    crime_w_release_date = release_date.merge(crime_label, on=['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'], how='outer')
    crime_w_release_date = crime_w_release_date.rename(columns={'num_of_felonies' : 'felony_counts_for_crime'})
    crime_w_release_date = crime_w_release_date.sort_values(['OFFENDER_NC_DOC_ID_NUMBER', 'release_date_with_imputation'])
    crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'])

    return crime_w_release_date


def create_time_to_next_incarceration_df(df):
    '''
    Creates a dataframe unique on OFFENDER_NC_DOC_ID_NUMBER and COMMITMENT_PREFIX (a person and a crime),
    and indicates the time since the person's last felony.

    Helper function for create_df_for_ml_pipeline
    '''
    df.sort_values(['OFFENDER_NC_DOC_ID_NUMBER', 'SENTENCE_EFFECTIVE(BEGIN)_DATE'])
    #arbitrarily large date
    df['start_time_of_next_incarceration'] = datetime.strptime('2080-01-01', '%Y-%m-%d')
    for index in range(0, df.shape[0] - 1):
        if df.loc[index, 'crime_felony_or_misd'] != 'FELON':
            continue
        else:
            if df.loc[index, 'OFFENDER_NC_DOC_ID_NUMBER'] == df.loc[index + 1, 'OFFENDER_NC_DOC_ID_NUMBER']:
                df.loc[index, 'start_time_of_next_incarceration'] = df.loc[index + 1, 'SENTENCE_EFFECTIVE(BEGIN)_DATE']

    return df


def create_recidvate_label(crime_w_release_date, recidviate_definition_in_days):
    crime_w_release_date['recidivate'] = 0
    diff = (crime_w_release_date['start_time_of_next_incarceration'] -
            crime_w_release_date['release_date_with_imputation'])
    crime_w_release_date.loc[diff < pd.to_timedelta(recidviate_definition_in_days, 'D'), 'recidivate'] = 1

    return crime_w_release_date


################################################################################
                        # ADD FEATURES
###############################################################################
def df_w_age_at_first_incarceration(input_df):
    '''
    Finds the age of the person at the first time they are arrested or incarercated.

    Inputs:
        - pandas df
            - crimes_w_demographic (pandas dataframe) merged  OFNT3AA1 and OFNT3CE1
    Output:
        - pandas dataframe - a series
    '''
    df = input_df.sort_values(['OFFENDER_NC_DOC_ID_NUMBER', 'SENTENCE_EFFECTIVE(BEGIN)_DATE']).copy()
    df['age_at_first_incarceration'] = (df['SENTENCE_EFFECTIVE(BEGIN)_DATE'] - pd.to_datetime(df['OFFENDER_BIRTH_DATE'])) / np.timedelta64(365, 'D')

    df_grouped = df.groupby(['OFFENDER_NC_DOC_ID_NUMBER']
                                    ).agg({'age_at_first_incarceration' : 'min'}
                                    ).reset_index()
    return input_df.merge(df_grouped, on='OFFENDER_NC_DOC_ID_NUMBER', how='left')


def make_count_vars_to_merge_onto_master_df(data, name_of_col):
    '''
    Takes a source dataframe and a column and returns a dataframe
    that just has the key identifying variables (DOC_ID and COMMITMENT_PREFIX),
    along with dummmy variables for that variable.

    Inputs:
        - data - (pandas dataframe) merged (or OFN....)

    Outputs:
        - (dataframe)
            - (primary key on 'OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX')
            - 'OFFENDER_NC_DOC_ID_NUMBER'
            - 'COMMITMENT_PREFIX'
            - bunch of dummies for the column (name_of_col)
    Next step: merge onto merged (or OFN....) by 'OFFENDER_NC_DOC_ID_NUMBER',
                            'COMMITMENT_PREFIX'
    '''
    to_add = pd.get_dummies(
            data,
            columns=[name_of_col]).groupby(
            ['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'],
            as_index=False).sum()
    filter_col = [col for col in to_add
                  if col.startswith(name_of_col + "_")]
    to_add = to_add[['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'] +
                    filter_col]

    return to_add


def merge_counts_variables(df, list_of_vars):
    doc_ids_to_keep = df['OFFENDER_NC_DOC_ID_NUMBER'].unique().tolist()
    subset_df = OFNT3CE1.loc[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].isin(doc_ids_to_keep),]

    for var in list_of_vars:
        print('\t\t\ton var ', var)
        to_add = make_count_vars_to_merge_onto_master_df(subset_df, var)
        df = df.merge(to_add, on=['OFFENDER_NC_DOC_ID_NUMBER',
                                              'COMMITMENT_PREFIX'], how='left')

    return df


def create_number_prev_incarcerations(df):
    df = df.sort_values(['OFFENDER_NC_DOC_ID_NUMBER',
                         'SENTENCE_EFFECTIVE(BEGIN)_DATE'])
    df['number_of_previous_incarcerations'] = 0
    nrows_df = df.shape[0]
    num_previous_incar = 0
    for i in range(1, nrows_df):
        if (df['OFFENDER_NC_DOC_ID_NUMBER'][i] ==
           df['OFFENDER_NC_DOC_ID_NUMBER'][i - 1]):
            num_previous_incar += 1
            df.loc[i, 'number_of_previous_incarcerations'] = num_previous_incar
        else:
            num_previous_incar = 0

    return df


print('\tfunctions loaded\t', datetime.now())
################################################################################
                            # SET GLOBALS
################################################################################
offender_filepath = "C:/Users/edwar.WJM-SONYLAPTOP/Desktop/ncdoc_data/data/preprocessed/OFNT3CE1.csv"
# offender_filepath = '/Users/bhargaviganesh/Documents/ncdoc_data/data/preprocessed/OFNT3CE1.csv'
inmate_filepath = "C:/Users/edwar.WJM-SONYLAPTOP/Desktop/ncdoc_data/data/preprocessed/INMT4BB1.csv"
# inmate_filepath = '/Users/bhargaviganesh/Documents/ncdoc_data/data/preprocessed/INMT4BB1.csv'
demographics_filepath = "C:/Users/edwar.WJM-SONYLAPTOP/Desktop/ncdoc_data/data/preprocessed/OFNT3AA1.csv"
# demographics_filepath = '/Users/bhargaviganesh/Documents/ncdoc_data/data/preprocessed/OFNT3AA1.csv'
begin_date = '2007-01-01'
end_date = '2018-01-01'

################################################################################
                        # SCRIPT - Merge and Format Data
################################################################################
OFNT3CE1 = clean_offender_data(offender_filepath)
print('\tOFNT3CE1 data cleaned and loaded\t', datetime.now())
INMT4BB1 = clean_inmate_data(inmate_filepath, begin_date, end_date)
print('\tINMT4BB1 data cleaned and loaded\t', datetime.now())
merged = merge_offender_inmate_df(OFNT3CE1, INMT4BB1)
print('\tOFNT3CE1 and INMT4BB1 merged\t', datetime.now())
crime_w_release_date = collapse_counts_to_crimes(merged, begin_date)
print("\t collapsed counts to crimes done\t", datetime.now())

df_to_ml_pipeline = crime_w_release_date.loc[crime_w_release_date['release_date_with_imputation'] > pd.to_datetime(begin_date)]
df_to_ml_pipeline = crime_w_release_date.loc[crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] < pd.to_datetime(end_date)]
df_to_ml_pipeline = df_to_ml_pipeline.reset_index()

#add recidivate label
crimes_w_time_since_release_date = create_time_to_next_incarceration_df(df_to_ml_pipeline)
crimes_w_recidviate_label= create_recidvate_label(crimes_w_time_since_release_date, 365)
# crimes_w_recidviate_label['recidivate'].describe()
print('\trecidivate label created\t', datetime.now())
OFNT3AA1 = load_demographic_data(demographics_filepath)
crimes_w_demographic = crimes_w_recidviate_label.merge(OFNT3AA1,
                        on='OFFENDER_NC_DOC_ID_NUMBER',
                        how='left')
print('\tdemographic data loaded and merged on\t', datetime.now())
#add age feature
crimes_w_demographic['age_at_crime'] = (crimes_w_demographic['SENTENCE_EFFECTIVE(BEGIN)_DATE'] -
                                pd.to_datetime(crimes_w_demographic['OFFENDER_BIRTH_DATE'])) / np.timedelta64(365, 'D')
crimes_w_demographic['years_in_prison'] = (crimes_w_demographic['release_date_with_imputation'] - \
            pd.to_datetime(crimes_w_demographic['SENTENCE_EFFECTIVE(BEGIN)_DATE'])) / np.timedelta64(365, 'D')
crimes_w_demographic = df_w_age_at_first_incarceration(crimes_w_demographic)
#Add variables for number of previous incarcerations
crimes_w_demographic = create_number_prev_incarcerations(crimes_w_demographic)
print('\ttime variables added\t', datetime.now())
# add count variables for attributes of each crime
list_of_vars_to_make_count_vars_with = ['COUNTY_OF_CONVICTION_CODE',
                                        'PUNISHMENT_TYPE_CODE',
                                        'COMPONENT_DISPOSITION_CODE',
                                        'PRIMARY_OFFENSE_CODE',
                                        'COURT_TYPE_CODE',
                                        'SENTENCING_PENALTY_CLASS_CODE']
final_df = merge_counts_variables(crimes_w_demographic,
                                  list_of_vars_to_make_count_vars_with)

final_df  = final_df.loc[final_df['crime_felony_or_misd']=='FELON',]
print('\tfinal dataset ready, about to run models\t', datetime.now())
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
models_to_run = ['SVM']
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
    'DT': {'criterion': ['gini'], 'max_depth': [1, 10, 20, 100], 'min_samples_split': [2, 5, 10]},
    'SVM': {'C': [0.01]},
    'KNN': {'n_neighbors': [25],'weights': ['uniform','distance'],'algorithm': ['ball_tree']},
    'GB': {'n_estimators': [10], 'learning_rate': [0.1,0.5], 'subsample': [0.1,0.5], 'max_depth': [5]},
    'BG': {'n_estimators': [10], 'max_samples': [.5]}}

test = {'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [2]}}

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
