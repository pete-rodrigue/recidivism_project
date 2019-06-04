import pandas as pd
import preprocess as pre
import ml_pipeline as ml
import preprocess_helper as rc


################################################################################
                              #SET GLOBALS FOR DATASET
################################################################################

infile = 'data/projects_2012_2013.csv'
outfile = 'output/final_submission_hw5.csv'
grid_size = 'test'

temporal_split_date_var = 'release_date_with_imputation'
testing_length, grace_period
validation_dates = ['2012-07-01', '2013-01-01', '2013-07-01']
# models_to_run = ['DT', 'RF', 'LR', 'GB', 'AB']
# did not run these models - 'SVM', 'KNN'
models_to_run = ['DT']

#inputs into the preprocess function - need to tell the function which variables to clean
columns_to_datetime = ['datefullyfunded', 'date_posted']
dummy_vars = ['teacher_prefix',
              'primary_focus_subject',
              'primary_focus_area', 'secondary_focus_subject',
              'secondary_focus_area', 'resource_type',
              'poverty_level', 'grade_level', 'school_metro', 'school_district',
              'school_county', 'school_city', 'school_state']

boolean_vars = ['eligible_double_your_impact_match', 'school_charter', 'school_magnet']
#some variables are id vars that we don't want to include these as features
vars_not_to_include = ["projectid", "projectid", "teacher_acctid", "schoolid", "school_ncesid"]
prediction_var = 'not_funded'


################################################################################
                              #SCRIPT
################################################################################
raw_data = rc.read_dataset(infile)
results_df =  pd.DataFrame(columns=('model_type', 'validation_date', 'clf', 'parameters', 'auc-roc', \
                                    'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50', \
                                    'r_at_1', 'r_at_2', "r_at_5", "r_at_10", 'r_at_20', 'r_at_30', 'r_at_50', \
                                    'f1_at_1', 'f1_at_2', "f1_at_5", "f1_at_10", 'f1_at_20', 'f1_at_30', 'f1_at_50',
                                     'baseline', 'len_x_train'))

clfs, grid = ml.define_clfs_params(grid_size)

for validation_date in validation_dates:
    train_set, validation_set = ml.temporal_split(raw_data, temporal_split_date_var, validation_date, 6, 60)

    #preprocess the train_set and test_set separately
    train_set = pre.pre_process(train_set, dummy_vars, boolean_vars, vars_not_to_include, columns_to_datetime)
    validation_set = pre.pre_process(validation_set, dummy_vars, boolean_vars, vars_not_to_include, columns_to_datetime)

    #create features - there will be features in the train that don't exist in test and vice versa
    #the model will only actually use the union of the two.
    train_features  = list(train_set.columns)
    test_features = list(validation_set.columns)

    #find union of the two lists
    intersection_features = list(set(train_features) & set(test_features))
    intersection_features.remove(prediction_var)

    #run the loop and save the output df
    results_df = ml.clf_loop(train_set, validation_set, intersection_features, prediction_var, models_to_run, clfs, grid, results_df, validation_date, outfile)
