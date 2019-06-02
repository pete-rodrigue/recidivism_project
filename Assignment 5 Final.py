#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import assignment3_functions as library
import pandas as pd
import numpy as np
from sklearn import (svm, ensemble, tree,
                     linear_model, neighbors, naive_bayes, dummy)
pd.options.display.max_columns = 999
import warnings 
warnings.simplefilter('ignore')
from datetime import date, datetime, timedelta


# ## Explore Data

# #### First, I load the data from Donors Choose

# In[3]:


donors_df = library.file_to_dataframe("projects_2012_2013.csv")


# #### Next, I create the target variable. I previously had done it backwards (labeling projects that did get funded within 60 days 1, and those that did not get funded 0, so here I have fixed that, because the prediction task is to determine the likelihood that a project will not be funded within 60 days

# In[4]:


donors_df['date_posted'] = pd.to_datetime(donors_df['date_posted'], format='%m/%d/%y')
donors_df['datefullyfunded'] = pd.to_datetime(donors_df['datefullyfunded'], format='%m/%d/%y')
donors_df['60_days_fullyfunded'] = np.where(donors_df['datefullyfunded'] - donors_df['date_posted'] <= pd.to_timedelta(60, unit='days'), 0, 1)
donors_df.head()


# #### Next, I look at my data to see which columns have NAs, and decide how to impute these values. In this case, I choose to just impute the continuous variable of students reached, during the pre-processing step.

# In[5]:


library.na_summary(donors_df)


# ## Pre-Processing

# #### Instead of hardcoding/dropping variables in place, I created a pre_processing function that takes lists of columns as its input, and then performs the necessary operations (imputing, dropping, etc). The parameters for that function are listed below. In constrast to my previous pipeline, the pre-processing step now occurs after the train, test split.

# In[6]:


to_dummy_list = ['school_magnet', 'school_charter', 'eligible_double_your_impact_match']
categorical_list = ['teacher_prefix', 'primary_focus_subject', 'primary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'school_district', 'secondary_focus_subject', 'secondary_focus_area', 'school_state', 'school_city', 'school_metro', 'school_county']
continuous_impute_list = ['students_reached']
vars_to_drop_all = ['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid', 'school_longitude', 'school_latitude']
vars_to_drop_dates = ['date_posted', 'datefullyfunded', '60_days_fullyfunded']


# ## Temporal Validation Analysis

# #### Below, I fixed the function that determines the dates for the 3 train, test sets. I introduced a grace period that can be specified (60 days in this case), so that 60 days can be used to evaluate the impact of the intervention.

# In[7]:


start_time_6mo = '2012-01-01'
end_time_6mo = '2013-12-31'
prediction_windows = [6]
temp_split_6mo = library.temporal_dates(start_time_6mo, end_time_6mo, prediction_windows, 60)
temp_split_6mo


# #### Below, I use the same parameter grid as before. I added in the Gradient Boost and Bagging Models, based on comments on my previous pipeline. I also found a way to make the training process go faster (introducing the n_jobs parameter), so I was able to run the analysis on the entire span of data.

# In[8]:


models_to_run = ['RF', 'AB', 'LR', 'KNN', 'SVM', 'DT', 'GB', 'BG']
 
classifiers = {'RF': ensemble.RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'LR': linear_model.LogisticRegression(penalty='l1', C=1e5, n_jobs=-1),
    'SVM': svm.LinearSVC(tol= 1e-5, random_state=0),
    'AB': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'DT': tree.DecisionTreeClassifier(),
    'KNN': neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    'GB': ensemble.GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
    'BG': ensemble.BaggingClassifier(linear_model.LogisticRegression(penalty='l1', C=1e5, n_jobs=-1))
        }

parameters = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5, 20, 100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,10,20,100],'min_samples_split': [2,5,10]},
    'SVM': {'C': [0.01]},
    'KNN': {'n_neighbors': [25],'weights': ['uniform','distance'],'algorithm': ['ball_tree']},
    'GB': {'n_estimators': [10], 'learning_rate': [0.1,0.5], 'subsample': [0.1,0.5], 'max_depth': [5]},
    'BG': {'n_estimators': [10], 'max_samples': [.5]}}


# #### I create a list of thresholds to loop over instead of hardcoding

# In[9]:


k_list = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]


# In[10]:


results_df, params = library.run_models(models_to_run, classifiers, parameters, donors_df, '60_days_fullyfunded', temp_split_6mo, 'date_posted', categorical_list, to_dummy_list, continuous_impute_list, vars_to_drop_all, vars_to_drop_dates, k_list, 'table_test_30.csv')


# #### After running through all the models, I produce a grid below that compares each model across various evaluation metrics (auc-roc, precision, recall, etc.) I added in f1 score, because it had been pointed out that I was missing that in my previous pipeline.

# In[11]:


results_df


# ## Model Evaluation

# #### After producing this grid, I identify the model with the best AUC-ROC score

# In[12]:


results_without_baseline = results_df[results_df['model_type'] != 'baseline']
best_model = results_without_baseline.loc[results_without_baseline['auc-roc'].idxmax()]
best_model


# #### Here are the parameters for the model with the best AUC/ROC curve

# In[13]:


params[92]


# #### Now, I train this model and produce the precision/recall curve

# In[18]:


x_train_log, x_test_log, y_train_log, y_test_log = library.temporal_split(donors_df, "date_posted" , "60_days_fullyfunded", best_model['train_start'], best_model['train_end'], best_model['test_start'], best_model['test_end'], vars_to_drop_dates)


# In[19]:


x_train_log, x_test_log, features_log = library.pre_process(x_train_log, x_test_log, categorical_list, to_dummy_list, continuous_impute_list, vars_to_drop_all)
x_train_log = x_train_log[features_log]
x_test_log = x_test_log[features_log]
best_logistic = best_model['classifier']
best_logistic


# In[20]:


y_pred_probs_best = best_logistic.fit(x_train_log, y_train_log).predict_proba(x_test_log)[:,1]


# In[22]:


library.plot_precision_recall_n(y_test_log, y_pred_probs_best, "Logistic Regression")


# #### Now I compare 5% precision, recall, and auc across models

# In[23]:


model_compare = results_without_baseline[['model_type', 'auc-roc', 'p_at_5', 'r_at_5']]
model_compare[['p_at_5', 'r_at_5']] = model_compare[['p_at_5', 'r_at_5']].apply(pd.to_numeric)
model_compare = model_compare.groupby('model_type').mean()
model_compare.reset_index(inplace=True)


# In[24]:


model_compare.plot(x='model_type', y=['auc-roc', 'p_at_5', 'r_at_5'], kind='bar')


# #### Finally, I compare the models over time, across the various evaluation metrics

# In[25]:


model_compare_time = results_without_baseline[['model_type', 'auc-roc', 'p_at_5', 'r_at_5', 'p_at_10', 'r_at_10', 'p_at_20', 'r_at_20', 'p_at_30', 'r_at_30', 'p_at_50', 'r_at_50', 'test_start']]
model_compare_time[['p_at_5', 'r_at_5', 'p_at_10', 'r_at_10', 'p_at_20', 'r_at_20', 'p_at_30', 'r_at_30', 'p_at_50', 'r_at_50']] = model_compare_time[['p_at_5', 'r_at_5', 'p_at_10', 'r_at_10', 'p_at_20', 'r_at_20', 'p_at_30', 'r_at_30', 'p_at_50', 'r_at_50']].apply(pd.to_numeric)
model_compare_time = model_compare_time.groupby(['model_type', 'test_start']).mean()
model_compare_time.reset_index(inplace=True)
model_compare_time

