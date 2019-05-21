#!/usr/bin/env python
# coding: utf-8

# Import packages
import pandas as pd
import numpy as np

# Import offenders dataset
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
# OFNT3CE1.loc[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER']=='0000006',].head(50)
# Only keep people that have ever had a felony offense
OFNT3CE1 = OFNT3CE1.groupby(
    'OFFENDER_NC_DOC_ID_NUMBER').filter(lambda x: x['has_felony'].max() == 1)

OFNT3CE1.shape  # Notice we have fewer rows now
# OFNT3CE1.head()
# OFNT3CE1['SENTENCE_EFFECTIVE(BEGIN)_DATE'].head(50)
# OFNT3CE1.dtypes

# Create a year variable with the year of the start of their sentence
# OFNT3CE1['year'] = OFNT3CE1[
#     'SENTENCE_EFFECTIVE(BEGIN)_DATE'].str.slice(0, 4).astype('int64')
# OFNT3CE1.head()
# # OFNT3CE1 = OFNT3CE1.loc[OFNT3CE1['year'] > 2000, :]
# OFNT3CE1 = OFNT3CE1.groupby(
#     'OFFENDER_NC_DOC_ID_NUMBER').filter(
#         lambda x: x['year'].max() > 2000)
# OFNT3CE1.shape
# OFNT3CE1.head()
# OFNT3CE1[['OFFENDER_NC_DOC_ID_NUMBER',
#           'year',
#           'PRIMARY_FELONY/MISDEMEANOR_CD.',
#           'SENTENCE_EFFECTIVE(BEGIN)_DATE',
#           'P&P_COMPONENT_STATUS_DATE',
#           'MINIMUM_SENTENCE_LENGTH',
#           'MAXIMUM_SENTENCE_LENGTH']].head(10)

# OFNT3CE1.columns
# OFNT3CE1.head(30)
# OFNT3CE1.loc[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == '0017471',
#                      ['OFFENDER_NC_DOC_ID_NUMBER',
#                       'COMMITMENT_PREFIX',
#                       'year', 'DATE_OFFENSE_COMMITTED_-_BEGIN',
#                       'PRIMARY_FELONY/MISDEMEANOR_CD.',
#                       'SENTENCE_EFFECTIVE(BEGIN)_DATE',
#                       'SENTENCE_COMPONENT_NUMBER',
#                       'PRIOR_RECORD_LEVEL_CODE']].head(20)

# dropping features we don't want to use:
OFNT3CE1 = OFNT3CE1.drop(['NC_GENERAL_STATUTE_NUMBER',
                          'LENGTH_OF_SUPERVISION',
                          'SUPERVISION_TERM_EXTENSION',
                          'SUPERVISION_TO_FOLLOW_INCAR.',
                          'G.S._MAXIMUM_SENTENCE_ALLOWED',
                          'ICC_JAIL_CREDITS_(IN_DAYS)'], axis=1)

#
#
#
# NOTE TO SELF: WE WANT TO MERGE ON THE DOC NUMBER AND THE COMMITMENT_PREFIX
# MAYBE ALSO THE COMPONENT NUMBER
#
INMT4BB1 = pd.read_csv(
    "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\ncdoc_data\\data\\preprocessed\\INMT4BB1.csv")
# INMT4BB1.columns
# INMT4BB1.head(10)
# INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 670272, :].head(20)
# INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 17471, :].head(20)
# INMT4BB1.head(50)
# INMT4BB1.PAROLE_SUPERVISION_BEGIN_DATE.unique()
# dropping features we don't want to use:
INMT4BB1 = INMT4BB1.drop(['INMATE_COMPUTATION_STATUS_FLAG',
                          'PROJECTED_RELEASE_DATE_(PRD)',
                          'PAROLE_DISCHARGE_DATE',
                          'PAROLE_SUPERVISION_BEGIN_DATE'], axis=1)

# INMT4BB1.head(10)
# OFNT3CE1.sort_values(['OFFENDER_NC_DOC_ID_NUMBER']).head(10)
# INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 34, :].head()
OFNT3CE1.dtypes
INMT4BB1.dtypes
# OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] = OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].astype('int64')
INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'] = pd.to_datetime(INMT4BB1['ACTUAL_SENTENCE_END_DATE'], errors='coerce')
# INMT4BB1[INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'].isnull() == True ].head(10)
# INMT4BB1 = INMT4BB1[ INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'].isnull() == False ]
INMT4BB1_subset = INMT4BB1.loc[(INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'] >= '2000-01-01') | (INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'].isnull() == True), :]
INMT4BB1_subset = INMT4BB1_subset.loc[(INMT4BB1_subset['clean_ACTUAL_SENTENCE_END_DATE'] < '2006-01-01') | (INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'].isnull() == True), :]

INMT4BB1_subset.shape

OFNT3CE1.applymap(np.isreal)
OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].astype('int')
OFNT3CE1.dtypes
INMT4BB1_subset.dtypes


merged = OFNT3CE1_subset.merge(INMT4BB1_subset,
                       left_on=['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'],
                       right_on=['INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX'],
                       how='right')

merged.head()
merged.loc[merged['INMATE_DOC_NUMBER'] == 34, :].head()

first_release = merged.sort_values("ACTUAL_SENTENCE_END_DATE").groupby("INMATE_DOC_NUMBER", as_index=False)['ACTUAL_SENTENCE_END_DATE'].first()

merged = merged.merge(first_release,
                      on='INMATE_DOC_NUMBER',
                      how='left')

merged = merged.rename(columns={"ACTUAL_SENTENCE_END_DATE_x": "ACTUAL_SENTENCE_END_DATE",
                       "ACTUAL_SENTENCE_END_DATE_y": "first_release_date"})
merged.head()

merged.dtypes
merged.columns
merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'] = pd.to_datetime(merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'], errors='coerce')
merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], errors='coerce')
merged['first_release_date'] = pd.to_datetime(merged['first_release_date'], errors='coerce')
merged['ACTUAL_SENTENCE_END_DATE'] = pd.to_datetime(merged['ACTUAL_SENTENCE_END_DATE'], errors='coerce')
merged.shape
merged = merged[merged['first_release_date'].between('1980-01-01', '2019-01-31')]
# merged[merged['ACTUAL_SENTENCE_END_DATE'].between('1980-01-01', '2019-01-31')].shape
# merged[merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'].between('1980-01-01', '2019-01-31')].shape
merged = merged[merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'].between('1980-01-01', '2019-01-31')]
# merged[merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'].between('1980-01-01', '2019-01-31')].shape
merged['first_release_date'].dtypes
merged['ACTUAL_SENTENCE_END_DATE'].dtypes
merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'].dtypes
merged['ACTUAL_SENTENCE_END_DATE'].head()
merged['time_elapsed'] = merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'] - merged['first_release_date']
merged['outcome'] = 0
merged.loc[(merged['time_elapsed'] >= '0 days') &( merged['time_elapsed'] <= '730 days'), 'outcome'] = 1


merged.head()

merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114, :]

merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114,
                      ['DATE_OFFENSE_COMMITTED_-_BEGIN',
                      'DATE_OFFENSE_COMMITTED_-_END',
                      'SENTENCE_BEGIN_DATE_(FOR_MAX)',
                      'ACTUAL_SENTENCE_END_DATE']]


merged_train = merged.loc[merged['']]







OFNT9BE1 = pd.read_csv(
    "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\ncdoc_data\\data\\preprocessed\\OFNT9BE1.csv")
