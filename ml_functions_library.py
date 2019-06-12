'''
Functions used for ml pipeline
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

def make_count_vars_to_merge_onto_master_df(data, name_of_col):
    '''
    Takes a source dataframe and a column name and returns a dataframe
    that just has the key identifying variables (DOC_ID and COMMITMENT_PREFIX),
    along with a count variable for the column of interest. Here we dummify
    variables outside of the merged data that are attributes of a crime, and
    need to be collapsed to the crime level to be added to the main dataset.
    The rest of the dummy variables are created after doing the train, test splits.

    Inputs:
        pandas dataframe collapsed by crime

    Returns:
        pandas dataframe with select counts variables added
    '''
    to_add = pd.get_dummies(
            data,
            columns=[name_of_col]).groupby(
            ['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'],
            as_index=False).sum()
    print('\t\t\tgot dummies')
    filter_col = [col for col in to_add
                  if col.startswith(name_of_col + "_")]
    to_add = to_add[['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'] +
                    filter_col]

    return to_add


def merge_counts_variables(df, source_df, list_of_vars):
    '''
    Takes a dataframe and a list of variables to merge and merges the counts
    variables above with the master dataframe.

    Inputs:
        df: master dataframe to merge on

    Returns:
        merged dataframe with counts variables
    '''

    doc_ids_to_keep = df['OFFENDER_NC_DOC_ID_NUMBER'].unique().tolist()
    subset_df = source_df.loc[source_df['OFFENDER_NC_DOC_ID_NUMBER'].isin(doc_ids_to_keep),]

    for var in list_of_vars:
        print('\t\t\ton var ', var)
        to_add = make_count_vars_to_merge_onto_master_df(subset_df, var)
        df = df.merge(to_add, on=['OFFENDER_NC_DOC_ID_NUMBER',
                                              'COMMITMENT_PREFIX'], how='left')
        print('\t\t\tdid merge for var', var)

    return df


def discretize(df, vars_to_discretize, num_bins=10):
    '''
    This function takes a dataframe and a list of variables
    to discretize and discretizes each continous variable.

    Inputs:
        pandas dataframe
        list of variables to discretize
        (optional) number of bins

    Returns:
        pandas dataframe with discretized variables
    '''
    for item in vars_to_discretize:
        new_label = item + '_discrete'
        df[new_label] = pd.qcut(df[item], num_bins)

    return df

def categorize(df, vars_to_categorize):
    '''
    This function takes a dataframe and a list of categorical variables
    and creates a binary/dummy variable from it

    Inputs:
        pandas dataframe
        list of variables to categorize

    Returns:
        pandas dataframe with dummy variables
    '''
    df_with_categorical = pd.get_dummies(df, columns=vars_to_categorize)

    return df_with_categorical


def cols_to_dummy(df, col_list, val):
    '''
    This function takes a dataframe, a list of columns, and
    a value that needs to be changed to a dummy. It returns
    a dataframe with this dummified column.

    Inputs:
        pandas dataframe
        list of columns to dummify
        value to change to 1

    Returns:
        pandas dataframe with dummy variables
    '''
    for col in col_list:
        df[col] = df[col].apply(lambda x: 1 if x == val else 0)

    return df


def impute_by(df, col, by='median'):
    '''
    Replace the NaNs with the column mean, median, or mode.
    Changes the column in place.

    Inputs:
        pandas dataframe
        column to impute
        method to impute by
    '''
    if by == 'median':
        df[col].fillna(df[col].median(), inplace=True)
    elif by == 'mode':
        df[col].fillna(df[col].mode(), inplace=True)
    elif by == 'zero':
        df[col].fillna(0, inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

def pre_process(train, test, categorical_list, continuous_impute_list, vars_to_drop, source_of_count_vars, counts_vars):
    '''
    This function takes a training set and a testing set and pre-processes columns in the dataset,
    to prepare them for the machine learning pipeline.
    The function takes a few lists of features to pre-process.

    Inputs:
        categorical_list: list of variables to make categorical
        to_dummy_list: list of variables to convert to dummies
        continious_impute_list: list of variables to impute
        vars_to_drop: list of variables to drop

    Returns:
        processed train and test sets and list of features in common between
        train and test sets
    '''
    features_train = set()
    features_test = set()
    final_features = set()
    processed_train = train.copy(deep=True)
    processed_test = test.copy(deep=True)
    processed_train.drop(vars_to_drop, axis=1, inplace=True)
    processed_test.drop(vars_to_drop, axis=1, inplace=True)
    final_processed_train = categorize(processed_train, categorical_list)
    final_processed_test = categorize(processed_test, categorical_list)
    final_processed_train = merge_counts_variables(final_processed_train, source_of_count_vars, counts_vars)
    final_processed_test = merge_counts_variables(final_processed_test, source_of_count_vars, counts_vars)
    for col in final_processed_train:
        if col in continuous_impute_list:
            impute_by(final_processed_train, col)
        features_train.add(col)
    for col in final_processed_test:
        if col in continuous_impute_list:
            impute_by(final_processed_test, col)
        features_test.add(col)
    final_features = list(features_train.intersection(features_test))

    return final_processed_train, final_processed_test, final_features

def generate_binary_at_k(y_pred_scores, k):
    '''
    This function converts probability scores into a binary outcome
    measure based on cutoff.

    Inputs:
        y_pred_scores: dataframe of predicted probabilites for y
        k: (float) threshold

    Returns:
        binary predictions
    '''
    cutoff_index = int(len(y_pred_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_pred_scores))]

    return test_predictions_binary

def evaluation_scores_at_k(y_test, y_pred_scores, k):
    '''
    This function uses sklearn's built in evaluation metrics
    to calculate precision, accuracy, recall for models, for
    a specified k threshold.

    Inputs:
        y_test: dataframe of true y values
        y_pred_scores: dataframe of predicted probabilites for y
        k: (float) threshold

    Returns:
        precision, accuracy, recall, f1 at k.
    '''
    df = pd.DataFrame()
    df['y_test'] = pd.Series(y_test)
    df['score'] = pd.Series(y_pred_scores)
    df['pred'] = 0
    cutoff = df['score'].quantile(1-k/100, interpolation='nearest')
    df.loc[df['score'] >= cutoff, 'pred' ] = 1
    df['tp'] = np.where((df['pred'] == 1) & (df['y_test'] == 1), 1, 0)
    df['tn'] = np.where((df['pred'] == 0) & (df['y_test'] == 0), 1, 0)
    precision_at_k = df['tp'].sum() / df['pred'].sum()
    actual_percent_of_population = df['pred'].mean()
    recall_at_k = df['tp'].sum()/df['y_test'].sum()
    accuracy_at_k = (df['tp'].sum() + df['tn'].sum())/len(y_test)
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    return precision_at_k, accuracy_at_k, recall_at_k, f1_at_k, actual_percent_of_population


def joint_sort_descending(l1, l2):
    '''
    Code adapted from Rayid Ghani's ml_functions in magic loop.
    This function sorts y_test and y_pred in descending order of probability.

    Inputs:
        l1: list 1
        l2: list 2

    Returns:
        sorted lists
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def plot_precision_recall_n(y_test, y_pred_scores, model_name):
    '''
    Code adapted from Rayid Ghani's ml_functions in magic loop.
    This function plots precision-recall curve for a given model.

    Inputs:
        y_test: true y values
        y_pred_scores: dataframe of predicted y values
        model_name: title for chart, model name
    '''
    y_score = y_pred_scores
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_scores)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_pred_scores[y_pred_scores>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    name = model_name
    plt.title(name)
    plt.show()
    plt.savefig('plots/precision_recall.png')

def temporal_dates(start_time, end_time, prediction_windows, grace_period=None):
    '''
    Adapted from Rayid's magic loops repository. This function takes
    a start time, end time, prediction window, and a grace period (time to assess evaluation)
    as arguments and returns a list of lists of the time splits.

    Inputs:
        start_time: date of the form Y-M-D
        end_time: date of the form Y-M-D
        prediction_windows:

    Returns:
        list of lists of datetime objects
    '''
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

    temp_dates = []

    for prediction_window in prediction_windows:
        windows = 1
        test_end_time = start_time_date
        while (end_time_date >= test_end_time + relativedelta(months=+prediction_window)):
            if grace_period:
                train_start_time = start_time_date
                train_end_time = train_start_time + windows * relativedelta(months=+prediction_window) - relativedelta(days=+1) - relativedelta(days=+grace_period+1)
                test_start_time = train_end_time + relativedelta(days=+1) + relativedelta(days=+grace_period+1)
                test_end_time = test_start_time + relativedelta(months=+prediction_window) - relativedelta(days=+1) - relativedelta(days=+grace_period+1)
                temp_dates.append([train_start_time,train_end_time,test_start_time,test_end_time,prediction_window])
                windows += 1
            else:
                train_start_time = start_time_date
                train_end_time = train_start_time + windows * relativedelta(months=+prediction_window) - relativedelta(days=+1)
                test_start_time = train_end_time + relativedelta(days=+1)
                test_end_time = test_start_time  + relativedelta(months=+prediction_window) - relativedelta(days=+1)
                temp_dates.append([train_start_time,train_end_time,test_start_time,test_end_time,prediction_window])
                windows += 1


    return temp_dates


def split_data(df, selected_y, selected_features):
    '''
    This function takes a dataframe, a list of selected features,
    a selected y variable, and a test size, and returns a
    training set and a testing set of the data.

    Inputs:
        pandas dataframe
        list of selected x variables
        selected y variable

    Returns:
        x-variable training dataset, y-variable training dataset,
        x-variable testing dataset,  y-variable testing dataset
    '''
    x = df[selected_features]
    y = df[selected_y]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y)

    return x_train, x_test, y_train, y_test


def temporal_split(df, time_var, selected_y, train_start, train_end, test_start, test_end, vars_to_drop_dates):
    '''
    This function takes a dataframe and splits it into training and test
    sets depending on the starting and end times provided.

    Inputs:
        df: pandas dataframe of interest
        time_var: variable in dataset representing time
        selected_y: variable to predict
        train_start: starting time for train set
        train_end: ending time for train set
        test_start: starting time for test set
        test_end: ending time for test set
        vars_to_drop_dates: date variables to drop

    Returns:
        x_train, x_test, y_train, y_test: train/test splits
    '''
    vars_to_drop_dates.append(selected_y)
    train_data = df[(df[time_var] >= train_start) & (df[time_var] <= train_end)]
    train_data.drop([time_var], axis=1)
    y_train = train_data[selected_y]
    x_train = train_data.drop(vars_to_drop_dates, axis=1)
    test_data = df[(df[time_var] >= test_start) & (df[time_var] <= test_end)]
    test_data.drop([time_var], axis=1)
    y_test = test_data[selected_y]
    x_test = test_data.drop(vars_to_drop_dates, axis=1)

    return x_train, x_test, y_train, y_test


def evaluation_metrics(k_list, y_test_sorted, y_pred_probs_sorted):
    '''
    This function takes a list of k values (percent of population) and
    returns a list of precision, accuracy, recall, and f1 values at
    those k values.
    '''
    full_list = []
    for k in k_list:
        precision, accuracy, recall, f1, actual_percent = evaluation_scores_at_k(y_test_sorted, y_pred_probs_sorted, k)
        full_list.append(precision)
        full_list.append(accuracy)
        full_list.append(recall)
        full_list.append(f1)
        full_list.append(actual_percent)

    return full_list


def run_models(models_to_run, classifiers, parameters, df, selected_y,
               temp_split, time_var, categorical_list,
               continuous_impute_list, vars_to_drop, vars_to_drop_dates,
               k_list, outfile, source_of_count_vars, counts_vars, export_data=False):
    '''
    Adapted from Rayid's magic loops repository.
    This function loops through all the models and classifiers and produces a grid with evaluation metrics at 1%, 2%, 5%, 10%, 20%, 30%, and 50% of the population.

    Inputs:
        models_to_run: list of models to run
        classifiers: classifiers to use
        parameters: parameter grid to use
        df: pandas dataframe with full data
        selected_y: variable to predict
        temp_split: dates to split on
        time_var: time variable to use
        categorical_list: list of variables to categorize
        to_dummy_list: list of variables to dummify
        continiuous_impute_list: list of variables to impute
        vars_to_drop: list of variables to drop
        vars_to_drop_dates: list of date variables to drop
        k_list: list of k values (percent of population) to calculate eval metrics for
        outfile: filename for output of final grid

    Returns:
        results_df: dataframe (grid) of models and evaluation metrics
        params: model parameters
    '''
    results_df = pd.DataFrame(columns=('train_start', 'train_end', 'test_start', 'test_end', 'model_type', 'classifier', 'parameter', 'train_size', 'test_size', 'auc-roc',
        'p_at_1', 'a_at_1', 'r_at_1', 'f1_at_1', 'perc_pop_at_1',
        'p_at_2', 'a_at_2', 'r_at_2', 'f1_at_2', 'perc_pop_at_2',
        'p_at_5', 'a_at_5', 'r_at_5', 'f1_at_5', 'perc_pop_at_5',
        'p_at_10', 'a_at_10', 'r_at_10', 'f1_at_10', 'perc_pop_at_10',
        'p_at_20', 'a_at_20', 'r_at_20', 'f1_at_20', 'perc_pop_at_20',
        'p_at_30', 'a_at_30', 'r_at_30', 'f1_at_30', 'perc_pop_at_30',
        'p_at_50', 'a_at_50', 'r_at_50', 'f1_at_50', 'perc_pop_at_50' ))

    params = []
    i = 1
    for timeframe in temp_split:
        print('On time split ' + str(i) + ' of ' + str(len(temp_split)) + ' total time splits.')
        i += 1
        train_start, train_end, test_start, test_end = timeframe[0], timeframe[1], timeframe[2], timeframe[3]
        x_train, x_test, y_train, y_test = temporal_split(df, time_var, selected_y, train_start, train_end, test_start, test_end, vars_to_drop_dates)
        x_train, x_test, features = pre_process(x_train, x_test,
                                                categorical_list,
                                                continuous_impute_list, vars_to_drop,
                                                source_of_count_vars, counts_vars)
        x_train = x_train[features]
        x_test = x_test[features]
        for index, classifier in enumerate([classifiers[x] for x in models_to_run]):
                print("\tRunning through model {}...".format(models_to_run[index]))
                parameter_values = parameters[models_to_run[index]]
                for p in ParameterGrid(parameter_values):
                    params.append(p)
                    try:
                        classifier.set_params(**p)
                        if models_to_run[index] == 'SVM':
                            y_pred_probs = classifier.fit(x_train, y_train).decision_function(x_test)
                        else:
                            y_pred_probs = classifier.fit(x_train, y_train).predict_proba(x_test)[:,1]
                        if export_data and i == 2:
                            x_test.to_csv('looking_at_x_test.csv')
                            pd.Series(y_pred_probs).to_csv('looking_at_y_pred_probs.csv')
                            pd.Series(y_test).to_csv('looking_at_y_test.csv')
                        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True, key=lambda x: x[0]))
                        metric_list = evaluation_metrics(k_list, y_test_sorted, y_pred_probs_sorted)
                        results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end,
                                                           models_to_run[index],
                                                           classifier,
                                                           p,
                                                           y_train.shape[0], y_test.shape[0],
                                                           metrics.roc_auc_score(y_test_sorted, y_pred_probs_sorted),
                                                           metric_list[0], metric_list[1], metric_list[2], metric_list[3], metric_list[4],
                                                           metric_list[5], metric_list[6], metric_list[7],metric_list[8], metric_list[9],
                                                           metric_list[10], metric_list[11], metric_list[12], metric_list[13], metric_list[14],
                                                           metric_list[15], metric_list[16], metric_list[17], metric_list[18], metric_list[19],
                                                           metric_list[20], metric_list[21], metric_list[22], metric_list[23], metric_list[24],
                                                           metric_list[25], metric_list[26], metric_list[27],  metric_list[28], metric_list[29],
                                                           metric_list[30], metric_list[31], metric_list[32],  metric_list[33], metric_list[34]]

                    except IndexError as e:
                        print('Error:',e)
                        continue

        results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end, "baseline", '', '', '', '',
                        y_test.sum()/len(y_test), '', '', '', '', '', '', '', '', '', '', '', '', '','', '', '', '',
                        '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']


        results_df.to_csv(outfile)

    return results_df, params
