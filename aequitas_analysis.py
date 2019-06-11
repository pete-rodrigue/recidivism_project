import numpy as np
import pandas as pd
import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot

# Example data:
# df = pd.DataFrame({'label_value': [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#                    'score': [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
#                    'race': ['b', 'b', 'b', 'b', 'b', 'w', 'w', 'w', 'w', 'w']})
df1 = pd.read_csv("looking_at_x_test.csv")
df2 = pd.read_csv("looking_at_y_test.csv", header=None)
df3 = pd.read_csv("looking_at_y_pred_probs.csv", header=None)
df = pd.concat([df1, df2], axis=1)
df.rename(columns={1: 'label_value'}, inplace=True)
df = pd.concat([df, df3], axis=1)
df.rename(columns={1: 'score'}, inplace=True)
df.head()
df.drop(0, axis=1, inplace=True)
# code below from here: https://stackoverflow.com/questions/27275236/pandas-best-way-to-select-all-columns-whose-names-start-with-x
filter_col_1 = [col for col in df if 'RACE' in col]
filter_col_2 = [col for col in df if 'GENDER' in col]
filter_col_1
filter_col_2
df = pd.concat([df[['score', 'label_value']],
                df[filter_col_1].idxmax(axis=1),
                df[filter_col_2].idxmax(axis=1)], axis=1)
df.head()
df.rename(columns={0: 'race', 1: 'gender'}, inplace=True)
df['race'] = df['race'].str.replace('OFFENDER_RACE_CODE_', '', regex=False)
df['gender'] = df['gender'].str.replace('OFFENDER_GENDER_CODE_', '', regex=False)
df.shape
df.head()


df.loc[df['score'].duplicated(), 'score']
unique_scores = np.sort(df['score'].unique())[::-1]
unique_scores
nrows_of_df = df.shape[0]
df['precision_score'] = 0

# for index, given_score in enumerate(unique_scores):
#     temp_df = df.loc[df['score'] >= given_score]
#     temp_precision = temp_df['label_value'].mean()
#     df.loc[df['score'] == given_score, 'precision_score'] = temp_precision


for percentile in [.5, .7, .8, .85, .9, .95, .99]:
    cutoff = df['score'].quantile(percentile, interpolation='nearest')
    temp_df = df.loc[df['score'] >= cutoff, 'label_value']
    temp_precision = temp_df.mean()
    actual_percent = temp_df.shape[0] / df.shape[0]
    print('precision at ', percentile, ' is ', round(temp_precision, 4), ' actual percent: ', round(1 - actual_percent, 6))


cutoff = df['score'].quantile(.99, interpolation='nearest')
df.loc[df['score'] == cutoff, ['precision_score', 'score']]
df.loc[df['score'] >= cutoff].shape[0] / nrows_of_df

# check for duplicate scores here
cutoff = df['score'].quantile(.95, interpolation='nearest')
df['new_score'] = 0
df.loc[df['score'] >= cutoff, 'new_score'] = 1
df.drop('score', axis=1, inplace=True)
df.rename(columns={'new_score': 'score'}, inplace=True)
aq_palette = sns.diverging_palette(225, 35, n=2)
sns.countplot(x="race", hue="score", data=df, palette=aq_palette)
sns.countplot(x="gender", hue="score", data=df, palette=aq_palette)
sns.countplot(x="race", hue="label_value", data=df, palette=aq_palette)
sns.countplot(x="gender", hue="label_value", data=df, palette=aq_palette)


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
# (what share of the true positives did we mistakenly say were negative?)


# Visualizing multiple user-specified absolute group metrics across all population groups
p = aqp.plot_group_metric_all(xtab, metrics=['ppr','pprev','fnr','fpr'], ncols=4)
a = aqp.plot_group_metric_all(xtab, ncols=3)

# Disparity
b = Bias()
bdf = b.get_disparity_predefined_groups(xtab,
                                        original_df=df,
                                        ref_groups_dict={'race':'WHITE', 'gender': 'MALE'},
                                        alpha=0.05, mask_significance=False)
bdf
bdf.style
calculated_disparities = b.list_disparities(bdf)
disparity_significance = b.list_significance(bdf.style)
calculated_disparities
disparity_significance
bdf[['attribute_name', 'attribute_value'] +  calculated_disparities + disparity_significance]
aqp.plot_disparity(bdf, group_metric='fpr_disparity', attribute_name='race', significance_alpha=0)


hbdf = b.get_disparity_predefined_groups(xtab, original_df=df,
                                         ref_groups_dict={'race':'WHITE',
                                                          'gender': 'MALE'},
                                         alpha=0.5,
                                         mask_significance=False)

majority_bdf = b.get_disparity_major_group(xtab, original_df=df, mask_significance=False)
majority_bdf[['attribute_name', 'attribute_value'] +  calculated_disparities + disparity_significance]
tm_capped = aqp.plot_disparity_all(hbdf, attributes=['race'], metrics = 'all', significance_alpha=0.05)
tm_capped = aqp.plot_disparity_all(hbdf, attributes=['gender', 'race'], metrics = 'all', significance_alpha=0.05)

f = Fairness()
fdf = f.get_group_value_fairness(bdf)
parity_detrminations = f.list_parities(fdf)
fdf[['attribute_name', 'attribute_value'] + absolute_metrics + calculated_disparities + parity_detrminations].style
fg = aqp.plot_fairness_group_all(fdf, ncols=5, metrics = "all")
n_tm = aqp.plot_fairness_disparity_all(fdf, attributes=['race', 'gender'],
                                       significance_alpha=0.05)
