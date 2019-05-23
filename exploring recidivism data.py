#!/usr/bin/env python
# coding: utf-8

# Import packages
import pandas as pd
import numpy as np
import matplotlib as plt

#
#  OFNT3CE1
#
#
#
#
OFNT3CE1 = pd.read_csv(
    "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\ncdoc_data\\data\\preprocessed\\OFNT3CE1.csv",
    dtype={'OFFENDER_NC_DOC_ID_NUMBER': str,
           'MAXIMUM_SENTENCE_LENGTH': str,
           'SPLIT_SENTENCE_ACTIVE_TERM': str,
           'SENTENCE_TYPE_CODE.5': str,
           'PRIOR_P&P_COMMNT/COMPONENT_ID': str,
           'ORIGINAL_SENTENCE_AUDIT_CODE': str})

OFNT3CE1.shape  # Number of rows
pd.options.display.max_columns = 100  # Set the max number of col to display

# Create a variable that indicates felony offenses
OFNT3CE1['has_felony'] = np.where(
    OFNT3CE1['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0)
# Only keep people that have ever had a felony offense
OFNT3CE1 = OFNT3CE1.groupby(
    'OFFENDER_NC_DOC_ID_NUMBER').filter(lambda x: x['has_felony'].max() == 1)

OFNT3CE1.shape  # Notice we have fewer rows now

OFNT3CE1['clean_SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(
        OFNT3CE1['SENTENCE_EFFECTIVE(BEGIN)_DATE'], errors='coerce')


# dropping features we don't want to use:
OFNT3CE1 = OFNT3CE1.drop(['NC_GENERAL_STATUTE_NUMBER',
                          'LENGTH_OF_SUPERVISION',
                          'SUPERVISION_TERM_EXTENSION',
                          'SUPERVISION_TO_FOLLOW_INCAR.',
                          'G.S._MAXIMUM_SENTENCE_ALLOWED',
                          'ICC_JAIL_CREDITS_(IN_DAYS)'], axis=1)

#
#  INMT4BB1
#
#
#
#
INMT4BB1 = pd.read_csv(
    "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\" +
    "ncdoc_data\\data\\preprocessed\\INMT4BB1.csv")

# dropping features we don't want to use:
INMT4BB1 = INMT4BB1.drop(['INMATE_COMPUTATION_STATUS_FLAG',
                          'PAROLE_DISCHARGE_DATE',
                          'PAROLE_SUPERVISION_BEGIN_DATE'], axis=1)


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
    INMT4BB1['release_date_with_imputation'])

INMT4BB1['imputed_release_date_flag'] = np.where(
        INMT4BB1['clean_projected_release_date'].notnull() &
        INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'].isnull(), 1, 0)

INMT4BB1['release_after_2000'] = np.where(
    (INMT4BB1['release_date_with_imputation'] > '1999-12-31'), 1, 0)
INMT4BB1['release_before_2006'] = np.where(
    (INMT4BB1['release_date_with_imputation'] < '2006-01-01'), 1, 0)
INMT4BB1['in_time_window'] = 0
INMT4BB1.loc[(INMT4BB1['release_after_2000'] > 0) &
             (INMT4BB1['release_before_2006'] > 0), 'in_time_window'] = 1

# INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 62, :]
# Only keep people with releases in our time window
INMT4BB1 = INMT4BB1.groupby(
    'INMATE_DOC_NUMBER').filter(lambda x: x['in_time_window'].max() == 1)

INMT4BB1.tail(10)

# Number of remaining people
INMT4BB1['INMATE_DOC_NUMBER'].unique().shape


# Making one person's id a number so we can make them all numeric
OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].loc[
        OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == 'T153879'] = "-999"
OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] = pd.to_numeric(
        OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'])
# OFNT3CE1.dtypes
merged = OFNT3CE1.merge(INMT4BB1,
                        left_on=['OFFENDER_NC_DOC_ID_NUMBER',
                                 'COMMITMENT_PREFIX',
                                 'SENTENCE_COMPONENT_NUMBER'],
                        right_on=['INMATE_DOC_NUMBER',
                                  'INMATE_COMMITMENT_PREFIX',
                                  'INMATE_SENTENCE_COMPONENT'],
                        how='outer')


# Find the people who have been to jail that we want to keep
ids_to_keep = merged['INMATE_DOC_NUMBER'].unique()
ids_to_keep = ids_to_keep[~np.isnan(ids_to_keep)]
merged = merged.loc[merged['OFFENDER_NC_DOC_ID_NUMBER'].isin(
            list(ids_to_keep)), : ]

merged.head(17)

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
        merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], :].head()









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
