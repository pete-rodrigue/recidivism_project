'''
Vedika Ahuja, Bhargavi Ganesh, and Pete Rodrigue
Library of functions used for data exploration.
'''
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import (svm, ensemble, tree,
                     linear_model, neighbors, naive_bayes, dummy)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_curve
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import ParameterGrid

#plots code adapted from: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

def file_to_dataframe(filename):
    '''
    Takes a filename and returns a pandas dataframe.

    Input:
        filename

    Returns:
        pandas dataframe

    '''

    if os.path.exists(filename):
        return pd.read_csv(filename)

def na_summary(df):
    ''''
    Takes a dataframe and returns a table
    showing which columns have NAs.

    Input:
        pandas dataframe

    Returns:
        table with nas
    '''
    return df.isna().sum(axis=0)

def describe_data(df, vars_to_describe=None):
    '''
    This function describes the data, providing
    basic descriptive statistics such as min,
    max, median, mean, etc.

    Input:
        pandas dataframe
        (optional) list of variables to describe

    Returns:
        table with min, max, mean, median, etc
        for each column in the specified df
    '''
    if vars_to_describe:
        df = df[vars_to_describe]

    return df.describe()

def histograms(df, vars_to_describe=None):
    '''
    Function that plots histogram of every variable in df.

    Input:
        pandas dataframe
        (optional) list of variables to describe
    '''
    if vars_to_describe:
        df = df[vars_to_describe]

    plt.rcParams['figure.figsize'] = 16, 12
    df.hist()
    plt.show()

def correlations(df, vars_to_describe=None):
    '''
    This function takes a dataframe and returns
    a correlation matrix with the specified variables.

    Input:
        pandas df
        (optional) list of variables to describe
    '''
    if vars_to_describe:
        df = df[vars_to_describe]

    return df.corr()

def correlation_matrix(correlations):
    '''
    This function takes a correlation table
    and plots a correlation matrix.

    Input:
        correlations: correlation table
    '''
    plt.rcParams['figure.figsize'] = 10, 10
    names = correlations.columns
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(names),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=30, rotation_mode='anchor', ha='left')
    ax.set_yticklabels(names)
    plt.show()
    plt.savefig('output/plots/corr_matrix.png')

def pairplot(df, vars_to_describe):
    '''
    This function takes a dataframe and variables
    to describe and plots a pairplot showing the
    relationship between variables.

    Inputs:
        pandas dataframe
        (optional) list of variables to describe
    '''
    plt.rcParams['figure.figsize']=(20,10)
    sns.pairplot(df, vars=vars_to_describe, dropna=True, height=3.5)
    plt.show()

def boxplots(df, vars_to_describe=None):
    '''
    This function takes a dataframe and variables
    to describe and plots boxplots for all the columns
    in the df.

    Inputs:
        pandas dataframe
        (optional) list of variables to describe
    '''
    if vars_to_describe:
        df = df[vars_to_describe]

    plt.rcParams['figure.figsize'] = 16, 12
    df.plot(kind='box', subplots=True,
    layout=(5, math.ceil(len(df.columns)/5)),
    sharex=False, sharey=False)
    plt.show()

def identify_ol(df, vars_to_describe=None):
    '''
    This function takes a dataframe, and returns a table of outliers

    Inputs:
        pandas dataframe
        (optional) list of variables to describe

    Returns:
        pandas dataframe with outliers
    '''
    subset_df = df.copy(deep=True)
    if vars_to_describe:
        subset_df = subset_df[vars_to_describe]
    Q1 = subset_df.quantile(0.25)
    Q3 = subset_df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = \
    subset_df[((subset_df < (Q1 - 1.5 * IQR)) | \
    (subset_df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df_out

def col_aggregation(df, col_to_group):
    '''
    This function takes in a dataframe, a string for a column name to group by,
    and a name for the new index, and returns a summary table of values sorted
    by percent.
    '''
    summary_table = df.groupby(col_to_group).size().reset_index(name='count')
    summary_table['percent'] = summary_table['count']/len(df)

    return summary_table.sort_values(by='percent', ascending=False)
