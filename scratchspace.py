# import pandas as pd
# import numpy as np
# import matplotlib as plt
# import datetime
# import os
# import aequitas
#
# pd.options.display.max_columns = 100
# print([0]*4)
# pd.Series([0] * len([0]*4))
#
# print(datetime.datetime.now())
#
# os.chdir('C:/Users/edwar.WJM-SONYLAPTOP/Desktop/ncdoc_data')
#
# file_path = "data/preprocessed/OFNT3CE1.csv"
#
#
# OFNT3CE1 = pd.read_csv(file_path,
#     dtype={'OFFENDER_NC_DOC_ID_NUMBER': str,
#            'MAXIMUM_SENTENCE_LENGTH': str,
#            'SPLIT_SENTENCE_ACTIVE_TERM': str,
#            'SENTENCE_TYPE_CODE.5': str,
#            'PRIOR_P&P_COMMNT/COMPONENT_ID': str,
#            'ORIGINAL_SENTENCE_AUDIT_CODE': str})
#
# OFNT3CE1.shape  # Number of rows
# OFNT3CE1 = OFNT3CE1[:10000]
#
# pd.options.display.max_columns = 100  # Set the max number of col to display
# OFNT3CE1.head()
# # Only keep people that have ever had a felony offense
# # Create a variable that indicates felony offenses
# OFNT3CE1['has_felony'] = np.where(
#     OFNT3CE1['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0).copy()
# OFNT3CE1 = OFNT3CE1.groupby(
#     'OFFENDER_NC_DOC_ID_NUMBER').filter(lambda x: x['has_felony'].max() == 1)
# OFNT3CE1.shape  # Notice we have fewer rows now
# OFNT3CE1.head()
# #clean the dates
# OFNT3CE1['clean_SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(
#         OFNT3CE1['SENTENCE_EFFECTIVE(BEGIN)_DATE'], errors='coerce')
#
#
# # dropping features we don't want to use:
# OFNT3CE1 = OFNT3CE1.drop(['NC_GENERAL_STATUTE_NUMBER',
#                           'LENGTH_OF_SUPERVISION',
#                           'SUPERVISION_TERM_EXTENSION',
#                           'SUPERVISION_TO_FOLLOW_INCAR.',
#                           'G.S._MAXIMUM_SENTENCE_ALLOWED',
#                           'ICC_JAIL_CREDITS_(IN_DAYS)'], axis=1)
#
# # Making one person's id a number so we can make them all numeric
# OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].loc[
#         OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == 'T153879'] = "-999"
# OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] = pd.to_numeric(
#         OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'])
#
# OFNT3CE1.tail(30)
#
#
# # def create_spread_cols_from_data(data, name_of_col):
# #     working_df = data[['OFFENDER_NC_DOC_ID_NUMBER',
# #                                      'COMMITMENT_PREFIX',
# #                                      name_of_col]]
# #     working_df = working_df.groupby(
# #                 ['OFFENDER_NC_DOC_ID_NUMBER',
# #                  'COMMITMENT_PREFIX']).aggregate(
# #                  lambda tdf: tdf.unique().tolist())
# #     max_len = 0
# #     for row in range(working_df.shape[0]):
# #         current_list = working_df.iloc[row][name_of_col]
# #         current_len = len(current_list)
# #         if current_len > max_len:
# #             max_len = current_len
# #     list_of_columns = []
# #     for i in range(max_len):
# #         col_name = "county_of_conviction_" + str(i)
# #         list_of_columns.append(col_name)
# #     to_bind = pd.DataFrame(working_df[name_of_col]
# #                            .values
# #                            .tolist(), columns=list_of_columns)
# #     to_return = pd.concat([working_df
# #                           .reset_index(), to_bind],
# #                           axis=1, ignore_index=True)
# #     to_return.columns = ['OFFENDER_NC_DOC_ID_NUMBER',
# #                          'COMMITMENT_PREFIX'] + \
# #                         ['dropme'] + list_of_columns
# #     to_return = to_return.drop('dropme', axis=1)
# #
# #     return to_return
#
# # create_spread_cols_from_data(data=OFNT3CE1,
# #                              name_of_col='COUNTY_OF_CONVICTION_CODE')
# # create_spread_cols_from_data(data=OFNT3CE1,
# #                              name_of_col='PUNISHMENT_TYPE_CODE')
# # create_spread_cols_from_data(data=OFNT3CE1,
# #                              name_of_col='COMPONENT_DISPOSITION_CODE')
# # create_spread_cols_from_data(data=OFNT3CE1,
# #                              name_of_col='PRIMARY_OFFENSE_CODE')
# # create_spread_cols_from_data(data=OFNT3CE1,
# #                              name_of_col='COURT_TYPE_CODE')
# # create_spread_cols_from_data(data=OFNT3CE1,
# #                              name_of_col='SENTENCING_PENALTY_CLASS_CODE')
#
#
#
# def make_dummy_vars_to_merge_onto_main_df(data, name_of_col):
#
#     return pd.concat([data[['OFFENDER_NC_DOC_ID_NUMBER',
#                             'COMMITMENT_PREFIX']],
#                      pd.get_dummies(data[name_of_col])],
#                      axis=1, ignore_index=False).groupby(
#                      ['OFFENDER_NC_DOC_ID_NUMBER',
#                       'COMMITMENT_PREFIX']
#                      ).sum().reset_index()
#
#
# test = pd.concat([OFNT3CE1[['OFFENDER_NC_DOC_ID_NUMBER',
#                             'COMMITMENT_PREFIX']],
#                  pd.get_dummies(OFNT3CE1['COUNTY_OF_CONVICTION_CODE'])],
#                  axis=1, ignore_index=False).groupby(
#                  ['OFFENDER_NC_DOC_ID_NUMBER',
#                   'COMMITMENT_PREFIX']
#                  ).sum().reset_index()
# test.head(10)
# OFNT3CE1.head()
# OFNT3CE1[['OFFENDER_NC_DOC_ID_NUMBER',
#                         'COMMITMENT_PREFIX',
#                         'SENTENCE_COMPONENT_NUMBER']].head(10)
# make_dummy_vars_to_merge_onto_main_df(data=OFNT3CE1,
#                                       name_of_col='COUNTY_OF_CONVICTION_CODE').tail(9)
# make_dummy_vars_to_merge_onto_main_df(data=OFNT3CE1,
#                                       name_of_col='PUNISHMENT_TYPE_CODE')
# make_dummy_vars_to_merge_onto_main_df(data=OFNT3CE1,
#                                       name_of_col='COMPONENT_DISPOSITION_CODE')
# make_dummy_vars_to_merge_onto_main_df(data=OFNT3CE1,
#                                       name_of_col='PRIMARY_OFFENSE_CODE').head()
# make_dummy_vars_to_merge_onto_main_df(data=OFNT3CE1,
#                                       name_of_col='COURT_TYPE_CODE').head()
# make_dummy_vars_to_merge_onto_main_df(data=OFNT3CE1,
#                                       name_of_col='SENTENCING_PENALTY_CLASS_CODE').head()
#
#
# file_path = "data/preprocessed/INMT9CF1.csv"
# INMT9CF1 = pd.read_csv(file_path)
#
# INMT9CF1.head()


import pickle
import pandas as pd
import os
import numpy as np

os.getcwd()
os.chdir('C:\\Users\\edwar.WJM-SONYLAPTOP\\Documents\\GitHub\\recidivism_project')
data = pd.read_pickle('pickled_final_df.pkl')

data.columns
data.shape


test = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
test
test.to_pickle('test_pickle.pkl')
test = None
test = pd.read_pickle('test_pickle.pkl')
test
