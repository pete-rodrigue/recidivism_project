#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

OFNT3CE1 = pd.read_csv(
    "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\ncdoc_data\\data\\preprocessed\\OFNT3CE1.csv",
    dtype={'OFFENDER_NC_DOC_ID_NUMBER': str,
           'MAXIMUM_SENTENCE_LENGTH': str,
           'SPLIT_SENTENCE_ACTIVE_TERM': str,
           'SENTENCE_TYPE_CODE.5': str,
           'PRIOR_P&P_COMMNT/COMPONENT_ID': str,
           'ORIGINAL_SENTENCE_AUDIT_CODE': str})

OFNT3CE1.shape
pd.options.display.max_columns = 100


# for index, value in enumerate(OFNT3CE1.columns):
#     print(index, value)

OFNT3CE1['has_felony'] = np.where(
    OFNT3CE1['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0)
OFNT3CE1 = OFNT3CE1.groupby(
    'OFFENDER_NC_DOC_ID_NUMBER').filter(lambda x: x['has_felony'].max() == 1)
OFNT3CE1.shape
OFNT3CE1.head()
OFNT3CE1['SENTENCE_EFFECTIVE(BEGIN)_DATE'].head(50)
OFNT3CE1.dtypes
OFNT3CE1['year'] = OFNT3CE1[
    'SENTENCE_EFFECTIVE(BEGIN)_DATE'].str.slice(0, 4).astype('int64')
OFNT3CE1.head()
# OFNT3CE1 = OFNT3CE1.loc[OFNT3CE1['year'] > 2000, :]
OFNT3CE1 = OFNT3CE1.groupby(
    'OFFENDER_NC_DOC_ID_NUMBER').filter(
        lambda x: x['year'].max() > 2000)
OFNT3CE1.shape
OFNT3CE1.head()
OFNT3CE1[['OFFENDER_NC_DOC_ID_NUMBER',
          'year',
          'PRIMARY_FELONY/MISDEMEANOR_CD.',
          'SENTENCE_EFFECTIVE(BEGIN)_DATE',
          'P&P_COMPONENT_STATUS_DATE',
          'MINIMUM_SENTENCE_LENGTH',
          'MAXIMUM_SENTENCE_LENGTH']].head(10)

OFNT3CE1.columns
OFNT3CE1.head(30)
OFNT3CE1.loc[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == '0017471',
                     ['OFFENDER_NC_DOC_ID_NUMBER',
                      'COMMITMENT_PREFIX',
                      'year', 'DATE_OFFENSE_COMMITTED_-_BEGIN',
                      'PRIMARY_FELONY/MISDEMEANOR_CD.',
                      'SENTENCE_EFFECTIVE(BEGIN)_DATE',
                      'SENTENCE_COMPONENT_NUMBER',
                      'PRIOR_RECORD_LEVEL_CODE']].head(20)

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
INMT4BB1.columns
INMT4BB1.head(10)
INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 17471, :].head(20)
INMT4BB1.head(50)
INMT4BB1.PAROLE_SUPERVISION_BEGIN_DATE.unique()
# dropping features we don't want to use:
INMT4BB1 = INMT4BB1.drop(['INMATE_COMPUTATION_STATUS_FLAG',
                          'PROJECTED_RELEASE_DATE_(PRD)',
                          'PAROLE_DISCHARGE_DATE',
                          'PAROLE_SUPERVISION_BEGIN_DATE'], axis=1)

INMT4BB1.head(50)
OFNT3CE1.head()
OFNT3CE1.dtypes
INMT4BB1.dtypes
OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] = OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].astype('int64')

INMT4BB1['ACTUAL_SENTENCE_END_DATE'] = pd.to_datetime(INMT4BB1['ACTUAL_SENTENCE_END_DATE'], errors='ignore')

INMT4BB1_subset = INMT4BB1.loc[INMT4BB1['ACTUAL_SENTENCE_END_DATE'] >= '2000-01-01', :]
INMT4BB1_subset = INMT4BB1_subset.loc[INMT4BB1_subset['ACTUAL_SENTENCE_END_DATE'] < '2006-01-01', :]
INMT4BB1_subset = INMT4BB1_subset.loc[INMT4BB1_subset['INMATE_SENTENCE_COMPONENT'] == 1, :]
INMT4BB1_subset.sort_values('INMATE_DOC_NUMBER').head(30)
INMT4BB1_subset.shape


OFNT3CE1_subset = OFNT3CE1.loc[OFNT3CE1['SENTENCE_COMPONENT_NUMBER'] == 1, :]
OFNT3CE1_subset.shape


OFNT3CE1_subset['OFFENDER_NC_DOC_ID_NUMBER'].dtypes
INMT4BB1_subset['INMATE_DOC_NUMBER'].dtypes
merged = OFNT3CE1_subset.merge(INMT4BB1_subset,
                       left_on='OFFENDER_NC_DOC_ID_NUMBER',
                       right_on='INMATE_DOC_NUMBER',
                       how='right')


merged.head()










OFNT9BE1 = pd.read_csv(
    "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\ncdoc_data\\data\\preprocessed\\OFNT9BE1.csv")
