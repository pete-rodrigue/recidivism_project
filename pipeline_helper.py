from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import preprocess_helper as rc
import preprocess as pre
from datetime import timedelta
from datetime import datetime
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys


def define_clfs_params(grid_size):
    """
    Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(),
        'ET': ExtraTreesClassifier(),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(),),
        'LR': LogisticRegression(),
        'SVM': svm.SVC(),
        'GB': GradientBoostingClassifier(),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(),
        'KNN': KNeighborsClassifier()
            }

    small_grid = {
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']}
           }


    test_grid = {
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0


def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    '''
    Classifies the prediction as 1 or 0 given the k threshold.
    The threshold is actually a percentile (?)
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)

    return precision


def recall_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)

    return recall

def f1_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    f1 = f1_score(y_true_sorted, preds_at_k)

    return f1

def create_evaluation_scores(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true_sorted, preds_at_k)
    f1 = f1_score(y_true_sorted, preds_at_k)

    return [precision, recall, f1]

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
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
    plt.title(name)
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()


def temporal_split(data, date_variable, validation_start_date, testing_length, grace_period):
    '''
    Creates a temporal split of the dataframe.

    Inputs:
        - data: pandas dataframe of the full dataset
        - date_variable: variable of the relevant date in the dataframe
        - training_start_date: date ('%Y-%m-%d') the date of the training window begins
        - training_length: (months) how long the training window is
        - testing_length: (months) how long the testing window is
        - grace_period: (days) the time period between the training and testing sets
                            - in this case, the grace period will come out of the
                            trainging set. For example, if the training set is 6 months,
                            and the grace period is 60 days, we will only consider
                            projects posted in the first 4 months of the
                            training window


    Outputs:
        - training_df: training dataset
        - testing_df: validation dataset

    '''
    # validation_start_date = datetime.strptime(validation_start_date, '%Y-%m-%d')
    # data[date_variable] = pd.to_datetime(data[date_variable])
    train_set = data.loc[data[date_variable] <= validation_start_date - timedelta(days=60)]
    #create validation set
    validation_end_date = validation_start_date + pd.DateOffset(months=testing_length) - timedelta(days=grace_period)
    validation_set = data.loc[(data[date_variable] > validation_start_date) & (data[date_variable] <= validation_end_date)]
    # validation_set = data.loc[data[date_variable] <= validation_end_date]

    return train_set, validation_set

####
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
        list of lists of datetime objects:
        train_start, train_end, test_start, test_end = timeframe[0], timeframe[1], timeframe[2], timeframe[3]
    '''
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

    temp_dates = []

    for prediction_window in prediction_windows:
        windows = 1
        test_end_time = start_time_date
        while (end_time_date >= test_end_time + relativedelta(months=+prediction_window)):
            train_start_time = start_time_date
            train_end_time = train_start_time + windows * relativedelta(months=+prediction_window) - relativedelta(days=+1) - relativedelta(days=+grace_period+1)
            test_start_time = train_end_time + relativedelta(days=+1) + relativedelta(days=+grace_period+1)
            test_end_time = test_start_time  + relativedelta(months=+prediction_window) - relativedelta(days=+1) - relativedelta(days=+grace_period+1)
            temp_dates.append([train_start_time,train_end_time,test_start_time,test_end_time,prediction_window])
            windows += 1

    return temp_dates

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
    train_data = df[(df[time_var] >= train_start) & (df[time_var] <= train_end)]
    # train_data.drop([time_var], axis=1)
    y_train = train_data[selected_y]
    x_train = train_data.drop(selected_y, axis=1)
    test_data = df[(df[time_var] >= test_start) & (df[time_var] <= test_end)]
    # test_data.drop([time_var], axis=1)
    y_test = test_data[selected_y]
    x_test = test_data.drop(selected_y, axis=1)

    return x_train, x_test, y_train, y_test


def clf_loop(train_set, validation_set, features, pred_var, models_to_run, clfs, grid, results_df, validation_date, csv_to_output):
    """
    Runs the loop using models_to_run, clfs, gridm and the data
    """
    X_train = train_set[features]
    y_train = train_set[pred_var]
    X_test = validation_set[features]
    y_test = validation_set[pred_var]

    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                # you can also store the model, feature importances, and prediction scores
                # we're only storing the metrics for now
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                baseline = y_pred_probs.sum()/len(y_pred_probs)
                results_df.loc[len(results_df)] = [models_to_run[index],validation_date, clf, p,
                                                   roc_auc_score(y_test, y_pred_probs),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                   baseline, len(X_train)]

                print("looping through", models_to_run[index], validation_date, clf, p)
                results_df.to_csv(csv_to_output, index=False)

            except IndexError as e:
                print('Error:',e)
                continue

        print(models_to_run[index], validation_date, clf, p, "Reading to file")
        # csv_to_output = outfile + models_to_run[index] + ".csv"
    results_df.to_csv(csv_to_output, index=False)

    return results_df

#plot the preicsion recall curves for the top model


def run_models(models_to_run, classifiers, parameters, df, selected_y, temp_split, time_var, categorical_list, to_dummy_list, continuous_impute_list, vars_to_drop, vars_to_drop_dates, k_list, outfile):
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
    results_df = pd.DataFrame(columns=('train_start', 'train_end', 'test_start', 'test_end', 'model_type', 'classifier', 'train_size', 'test_size', 'auc-roc',
        'p_at_1', 'a_at_1', 'r_at_1', 'f1_at_1', 'p_at_2', 'a_at_2', 'r_at_2', 'f1_at_2', 'p_at_5', 'a_at_5', 'r_at_5', 'f1_at_5', 'p_at_10', 'a_at_10', 'r_at_10', 'f1_at_10',
        'p_at_20', 'a_at_20', 'r_at_20', 'f1_at_20', 'p_at_30', 'a_at_30', 'r_at_30', 'f1_at_30', 'p_at_50', 'a_at_50', 'r_at_50', 'f1_at_50'))

    params = []

    for timeframe in temp_split:
        train_start, train_end, test_start, test_end = timeframe[0], timeframe[1], timeframe[2], timeframe[3]
        x_train, x_test, y_train, y_test = temporal_split(df, time_var, selected_y, train_start, train_end, test_start, test_end, vars_to_drop_dates)
        x_train, x_test, features = pre_process(x_train, x_test, categorical_list, to_dummy_list, continuous_impute_list, vars_to_drop)
        x_train = x_train[features]
        x_test = x_test[features]
        for index, classifier in enumerate([classifiers[x] for x in models_to_run]):
                print("Running through model {}...".format(models_to_run[index]))
                parameter_values = parameters[models_to_run[index]]
                for p in ParameterGrid(parameter_values):
                    params.append(p)
                    try:
                        classifier.set_params(**p)
                        if models_to_run[index] == 'SVM':
                            y_pred_probs = classifier.fit(x_train, y_train).decision_function(x_test)
                        else:
                            y_pred_probs = classifier.fit(x_train, y_train).predict_proba(x_test)[:,1]
                        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                        metric_list = evaluation_metrics(k_list, y_test_sorted, y_pred_probs_sorted)
                        results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end,
                                                           models_to_run[index],
                                                           classifier,
                                                           y_train.shape[0], y_test.shape[0],
                                                           metrics.roc_auc_score(y_test_sorted, y_pred_probs),
                                                           metric_list[0], metric_list[1], metric_list[2], metric_list[3],
                                                           metric_list[4], metric_list[5], metric_list[6], metric_list[7],
                                                           metric_list[8], metric_list[9], metric_list[10], metric_list[11],
                                                           metric_list[12], metric_list[13], metric_list[14], metric_list[15],
                                                           metric_list[16], metric_list[17], metric_list[18], metric_list[19],
                                                           metric_list[20], metric_list[21], metric_list[22], metric_list[23],
                                                           metric_list[24], metric_list[25], metric_list[26], metric_list[27]]

                    except IndexError as e:
                        print('Error:',e)
                        continue

        results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end, "baseline", '', '', '',
                        y_test.sum()/len(y_test), '', '', '', '', '', '', '', '', '', '', '', '', '','', '', '', '',
                        '', '', '', '', '', '', '', '', '', '', '']


    results_df.to_csv(outfile)

    return results_df, params
