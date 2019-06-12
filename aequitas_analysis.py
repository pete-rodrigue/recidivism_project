import numpy as np
import pandas as pd
import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot

# Load data from test split 1
df1 = pd.read_csv("looking_at_x_test.csv")
df2 = pd.read_csv("looking_at_y_test.csv", header=None)
df3 = pd.read_csv("looking_at_y_pred_probs.csv", header=None)
# Concatonate data together
df = pd.concat([df1, df2], axis=1)
df.rename(columns={1: 'label_value'}, inplace=True)
df = pd.concat([df, df3], axis=1)
df.rename(columns={1: 'score'}, inplace=True)
df.drop(0, axis=1, inplace=True)
# code below from here: https://stackoverflow.com/questions/27275236/pandas-best-way-to-select-all-columns-whose-names-start-with-x
filter_col_1 = [col for col in df if 'RACE' in col]
filter_col_2 = [col for col in df if 'GENDER' in col]
# Turn dummy variables back into categorical variables
df = pd.concat([df[['score', 'label_value']],
                df[filter_col_1].idxmax(axis=1),
                df[filter_col_2].idxmax(axis=1)], axis=1)
df.rename(columns={0: 'race', 1: 'gender'}, inplace=True)
df['race'] = df['race'].str.replace('OFFENDER_RACE_CODE_', '', regex=False)
df['gender'] = df['gender'].str.replace('OFFENDER_GENDER_CODE_', '', regex=False)
df.shape
df.head()

# See if we have duplicate scores:
df.loc[df['score'].duplicated(), 'score']
# Get unique scores:
unique_scores = np.sort(df['score'].unique())[::-1]
# We're going to make predicted labels based on our prediction score
# with a threshold of 5%
cutoff = df['score'].quantile(.95, interpolation='nearest')
df['new_score'] = 0
df.loc[df['score'] >= cutoff, 'new_score'] = 1
# We get about 5% of our people:
df['new_score'].describe()
df.drop('score', axis=1, inplace=True)
df.rename(columns={'new_score': 'score'}, inplace=True)
aq_palette = sns.diverging_palette(225, 35, n=2)
# Plot our predicted scores by race and gender:
sns.countplot(x="race", hue="score", data=df, palette=aq_palette)
sns.countplot(x="gender", hue="score", data=df, palette=aq_palette)
# Plot our actual labels by race and gender:
sns.countplot(x="race", hue="label_value", data=df, palette=aq_palette)
sns.countplot(x="gender", hue="label_value", data=df, palette=aq_palette)

df.groupby('gender').describe()
df.groupby('race').describe()

g = Group()
xtab, _ = g.get_crosstabs(df)
xtab  # Table of statistics by group
# View calculated absolute metrics for each sample population group
absolute_metrics = g.list_absolute_metrics(xtab)
# FNR: FN / (True Condition Negative) = FN / (FN + TP)
# (what share of the true positives did we mistakenly say were negative?)
# FOR: false omission rate: FN / (FN + TN)
# (how many of the people we labeled negative were false negatives (really postive)?)
# FDR: false discovery rate: FP / (FP + TP)
# (how many of the people we labeled positive were false positives?)
xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)
df['race'].value_counts() / df.shape[0]
aqp = Plot()
# Visualizing multiple user-specified absolute group metrics across all population groups
a = aqp.plot_group_metric_all(xtab, ncols=3)
a.savefig('fairness_plot.png')

# True prevalences
df.groupby('gender').describe()

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


hbdf = b.get_disparity_predefined_groups(xtab, original_df=df,
                                         ref_groups_dict={'race':'WHITE',
                                                          'gender': 'MALE'},
                                         alpha=0.5,
                                         mask_significance=False)

majority_bdf = b.get_disparity_major_group(xtab, original_df=df, mask_significance=False)
majority_bdf[['attribute_name', 'attribute_value'] +  calculated_disparities + disparity_significance]

tm_capped = aqp.plot_disparity_all(hbdf, attributes=['gender', 'race'], metrics = 'all', significance_alpha=0.05)

f = Fairness()
fdf = f.get_group_value_fairness(bdf)
parity_detrminations = f.list_parities(fdf)
fdf[['attribute_name', 'attribute_value'] + absolute_metrics + calculated_disparities + parity_detrminations].style
fg = aqp.plot_fairness_group_all(fdf, ncols=5, metrics = "all")
n_tm = aqp.plot_fairness_disparity_all(fdf, attributes=['race', 'gender'],
                                       significance_alpha=0.05)
