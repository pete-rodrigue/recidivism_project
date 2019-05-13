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
OFNT3CE1.loc[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == '0017471',
         ['OFFENDER_NC_DOC_ID_NUMBER',
          'year', 'DATE_OFFENSE_COMMITTED_-_BEGIN',
          'PRIMARY_FELONY/MISDEMEANOR_CD.',
          'SENTENCE_EFFECTIVE(BEGIN)_DATE',
          'P&P_COMPONENT_STATUS_DATE',
          'MINIMUM_SENTENCE_LENGTH',
          'MAXIMUM_SENTENCE_LENGTH']].head(20)
