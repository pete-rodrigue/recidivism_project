import pandas as pd
import numpy as np


offenders_file = pd.read_csv("data/preprocessed/OFNT3CE1.csv")
offenders_file['has_felony'] = np.where(offenders_file['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0)
offenders_smaller = offenders_file.groupby('OFFENDER_NC_DOC_ID_NUMBER').filter(lambda x: x['has_felony'].max() == 1)
offenders_smaller['sentence_begin_year'] = offenders_smaller['SENTENCE_EFFECTIVE(BEGIN)_DATE'].str.slice(0,4).astype('int64')
offenders_subset = offenders_smaller[offenders_smaller['sentence_begin_year'] > 2005]