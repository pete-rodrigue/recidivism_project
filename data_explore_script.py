'''
Vedika Ahuja, Bhargavi Ganesh, and Pete Rodrigue
Script which runs through basic data exploration.
'''
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import data_explore_library as dl
import processing_library as pl

################################################################################
                            # SET GLOBALS
################################################################################
offender_filepath = 'ncdoc_data/data/preprocessed/OFNT3CE1.csv'
inmate_filepath = 'ncdoc_data/data/preprocessed/INMT4BB1.csv'
demographics_filepath = 'ncdoc_data/data/preprocessed/OFNT3AA1.csv'
begin_date = '2007-01-01'
end_date = '2018-01-01'
recidivate_definition_in_days = 365
################################################################################
                    # SCRIPT - Load and Format Data
################################################################################
#Clean different tables and merge
final_df = pl.make_final_df(offender_filepath, inmate_filepath, demographics_filepath, begin_date, end_date, recidivate_definition_in_days)
#summarizing NAs
dl.na_summary(final_df)

vars_to_describe = ['total_counts_for_crime', 'felony_counts_for_crime', 'recidivate',
'OFFENDER_HEIGHT_(IN_INCHES)', 'OFFENDER_WEIGHT_(IN_LBS)', 'age_at_crime', 'years_in_prison',
'age_at_first_incarceration', 'number_of_previous_incarcerations']
describe = dl.describe_data(final_df, vars_to_describe=vars_to_describe)
describe.to_csv('output/plots/descriptive_table.csv')

dl.histograms(final_df, vars_to_describe=vars_to_describe)

correlations = dl.correlations(final_df, vars_to_describe=vars_to_describe)
dl.correlation_matrix(correlations)

dl.boxplots(final_df, vars_to_describe=vars_to_describe)

summary_table_race = dl.col_aggregation(final_df, 'OFFENDER_RACE_CODE')
summary_table_race.to_csv('output/plots/race_table.csv')

summary_table_gender = dl.col_aggregation(final_df, 'OFFENDER_GENDER_CODE')
summary_table_gender.to_csv('output/plots/gender_table.csv')

summary_table_ethnic = dl.col_aggregation(final_df, ['OFFENDER_RACE_CODE','OFFENDER_ETHNIC_CODE'])
summary_table_ethnic.to_csv('output/plots/race_ethnicity_table.csv')

summary_table_citizen = dl.col_aggregation(final_df, 'OFFENDER_CITIZENSHIP_CODE')
summary_table_citizen.to_csv('output/plots/citizen_table.csv')
