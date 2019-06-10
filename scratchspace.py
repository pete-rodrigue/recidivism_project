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

#
# import pickle
# import pandas as pd
# import os
# import numpy as np

# os.getcwd()
# os.chdir('C:\\Users\\edwar.WJM-SONYLAPTOP\\Documents\\GitHub\\recidivism_project')
# data = pd.read_pickle('pickled_final_df.pkl')
#
# data.columns
# data.shape
#
#
# test = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
# test
# test.to_pickle('test_pickle.pkl')
# test = None
# test = pd.read_pickle('test_pickle.pkl')
# test
#
# a = ['a', 'b', 2]
# with open("confusion_matrix_log.txt", "a+") as cm_log:
#     cm_log.write('\nHello again! cool\n' + str(a))

import pandas as pd
import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot

df = pd.DataFrame({'label_value': [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                   'score': [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                   'race': ['b', 'b', 'b', 'b', 'b', 'w', 'w', 'w', 'w', 'w']})
df


aq_palette = sns.diverging_palette(225, 35, n=2)
sns.countplot(x="race", hue="score", data=df, palette=aq_palette)
sns.countplot(x="race", hue="label_value", data=df, palette=aq_palette)


g = Group()
df.dtypes
xtab, _ = g.get_crosstabs(df)
xtab
absolute_metrics = g.list_absolute_metrics(xtab)
# View calculated counts across sample population groups
xtab[[col for col in xtab.columns if col not in absolute_metrics]]

# View calculated absolute metrics for each sample population group
# FOR: false omission rate: FN / (FN + TN)
# (how many of the people we labeled negative were false negatives (really postive)?)
# FDR: false discovery rate: FP / (FP + TP)
# (how many of the people we labeled positive were false positives?)
xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)

aqp = Plot()
fnr = aqp.plot_group_metric(xtab, 'fnr')
# FNR: FN / (True Condition Negative) = FN / (FN + TP)
# (how many of the true positives did we mistakenly say were negative?)
# So graph above says we said half the true positives for white ppl where
# mistakenly predicted negative.


# Visualizing multiple user-specified absolute group metrics across all population groups
p = aqp.plot_group_metric_all(xtab, metrics=['ppr','pprev','fnr','fpr'], ncols=4)
a = aqp.plot_group_metric_all(xtab, ncols=3)

# Disparity
b = Bias()
bdf = b.get_disparity_predefined_groups(xtab, original_df=df, ref_groups_dict={'race':'w'}, alpha=0.05, mask_significance=True)
bdf.style
calculated_disparities = b.list_disparities(bdf)
disparity_significance = b.list_significance(bdf)
calculated_disparities
disparity_significance
bdf[['attribute_name', 'attribute_value'] +  calculated_disparities + disparity_significance]
aqp.plot_disparity(bdf, group_metric='fpr_disparity', attribute_name='race', significance_alpha=0.2)


hbdf = b.get_disparity_predefined_groups(xtab, original_df=df,
                                         ref_groups_dict={'race':'w'},
                                         alpha=0.5,
                                         mask_significance=False)
aqp.plot_disparity(hbdf, group_metric='fpr_disparity', attribute_name='race', significance_alpha=0.05)
majority_bdf = b.get_disparity_major_group(xtab, original_df=df, mask_significance=True)
majority_bdf[['attribute_name', 'attribute_value'] +  calculated_disparities + disparity_significance]
j = aqp.plot_disparity_all(majority_bdf, metrics=['precision_disparity'], significance_alpha=0.5)
tm_capped = aqp.plot_disparity_all(hbdf, attributes=['race'], metrics = 'all', significance_alpha=0.05)

f = Fairness()
fdf = f.get_group_value_fairness(bdf)
parity_detrminations = f.list_parities(fdf)
fdf[['attribute_name', 'attribute_value'] + absolute_metrics + calculated_disparities + parity_detrminations].style
fg = aqp.plot_fairness_group_all(fdf, ncols=5, metrics = "all")
n_tm = aqp.plot_fairness_disparity_all(fdf, attributes=['race'],
                                       significance_alpha=0.05)



a = zip(*sorted(zip([.7, .7, .7, .2], [.25, 1.5, 1, 0]), reverse=True, key=lambda x: x[0]))
for i in a:
    print(i)
