#!/usr/bin/env python
# coding: utf-8

# Import packages
import pandas as pd
import numpy as np
import matplotlib as plt
import datetime
pd.options.display.max_columns = 100

##Steps
#figure out sentence begin date
#figure out how to calculate the difference since the last release

#
################################################################################
                            # SET GLOBALS
################################################################################
offender_filepath = "ncdoc_data/data/preprocessed/OFNT3CE1.csv"
inmate_filepath = "ncdoc_data/data/preprocessed/INMT4BB1.csv"
begin_date = pd.to_datetime('2008-01-01')
end_date = pd.to_datetime('2019-12-31')
#note, we will have to start with the last end date possible before we collapse
#the counts by crime

################################################################################
                            # SCRIPT
################################################################################

OFNT3CE1 = clean_offender_data(offender_filepath)
OFNT3CE1.shape
INMT4BB1 = clean_inmate_data(inmate_filepath, begin_date, end_date)
INMT4BB1.shape
merged = merge_offender_inmate_df(OFNT3CE1, INMT4BB1)
merged.shape

crime_w_release_date = collapse_counts_to_crimes(merged, begin_date)
crime_w_release_date.loc[crime_w_release_date['OFFENDER_NC_DOC_ID_NUMBER']==1188286, :]
diff = crime_w_release_date['time_of_last_felony_release'] - crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE']

#I think this is the place to filter the data (this is specifically for the ml pipeline.
#will have to use different filters to add relevant features
df_to_ml_pipeline = crime_w_release_date.loc[crime_w_release_date['release_date_with_imputation'] > begin_date]
df_to_ml_pipeline = crime_w_release_date.loc[crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] < end_date]
df_to_ml_pipeline.shape
df_to_ml_pipeline.to_csv("trial_output.csv")
crimes_w_time_since_release_date = create_time_since_last_release_df_v2(crime_w_release_date)
crimes_w_recidviate_label = create_recidvate_label(crimes_w_time_since_release_date, 365)

#add features and run!!!
                    # READ AND CLEAN OFNT3CE1
################################################################################
def clean_offender_data(offender_filepath):
    '''
    Takes the offender dataset (OFNT3CE1), cleans it, and outputs it as a to_csv
    '''
    OFNT3CE1 = pd.read_csv(offender_filepath,
        dtype={'OFFENDER_NC_DOC_ID_NUMBER': str,
               'MAXIMUM_SENTENCE_LENGTH': str,
               'SPLIT_SENTENCE_ACTIVE_TERM': str,
               'SENTENCE_TYPE_CODE.5': str,
               'PRIOR_P&P_COMMNT/COMPONENT_ID': str,
               'ORIGINAL_SENTENCE_AUDIT_CODE': str})

    OFNT3CE1.shape  # Number of rows
    pd.options.display.max_columns = 100  # Set the max number of col to display

    # Only keep people that have ever had a felony offense
    # Create a variable that indicates felony offenses
    OFNT3CE1['has_felony'] = np.where(
        OFNT3CE1['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0).copy()
    OFNT3CE1 = OFNT3CE1.groupby(
        'OFFENDER_NC_DOC_ID_NUMBER').filter(lambda x: x['has_felony'].max() == 1)
    OFNT3CE1.shape  # Notice we have fewer rows now

    #clean the dates
    OFNT3CE1['clean_SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(
            OFNT3CE1['SENTENCE_EFFECTIVE(BEGIN)_DATE'], errors='coerce')


    # dropping features we don't want to use:
    OFNT3CE1 = OFNT3CE1.drop(['NC_GENERAL_STATUTE_NUMBER',
                              'LENGTH_OF_SUPERVISION',
                              'SUPERVISION_TERM_EXTENSION',
                              'SUPERVISION_TO_FOLLOW_INCAR.',
                              'G.S._MAXIMUM_SENTENCE_ALLOWED',
                              'ICC_JAIL_CREDITS_(IN_DAYS)'], axis=1)

    # Making one person's id a number so we can make them all numeric
    OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].loc[
            OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == 'T153879'] = "-999"
    OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] = pd.to_numeric(
            OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'])

    # OFNT3CE1.to_csv("OFNT3CE1.csv")
    return OFNT3CE1


################################################################################
                        # CLEAN AND READ  INMT4BB1
################################################################################
def clean_inmate_data(inmate_filepath, begin_date, end_date):
    '''
    Reads and cleans the inmate data.

    Inputs:
        - inmate_filepath: csv file path
        - begin_date: The beginning date of the time period of interest -
                        This is the release date we want to look at
        - end_date: The end date of the time period
    '''
    INMT4BB1 = pd.read_csv(inmate_filepath)
    INMT4BB1.head()

    # dropping features we don't want to use:
    INMT4BB1 = INMT4BB1.drop(['INMATE_COMPUTATION_STATUS_FLAG',
                              'PAROLE_DISCHARGE_DATE',
                              'PAROLE_SUPERVISION_BEGIN_DATE'], axis=1)
    #clean dates
    INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'] = pd.to_datetime(
                INMT4BB1['ACTUAL_SENTENCE_END_DATE'], errors='coerce')
    INMT4BB1['clean_projected_release_date'] = pd.to_datetime(
                INMT4BB1['PROJECTED_RELEASE_DATE_(PRD)'], errors='coerce')
    INMT4BB1['clean_SENTENCE_BEGIN_DATE_(FOR_MAX)'] = pd.to_datetime(
            INMT4BB1['SENTENCE_BEGIN_DATE_(FOR_MAX)'], errors='coerce')

    INMT4BB1['release_date_with_imputation'] = INMT4BB1[
                                'clean_ACTUAL_SENTENCE_END_DATE']
    INMT4BB1.head()
    INMT4BB1['release_date_with_imputation'] = np.where(
        (INMT4BB1['release_date_with_imputation'].isnull()),
        INMT4BB1['clean_projected_release_date'],
        INMT4BB1['release_date_with_imputation']).copy()

    INMT4BB1['imputed_release_date_flag'] = np.where(
            INMT4BB1['clean_projected_release_date'].notnull() &
            INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'].isnull(), 1, 0).copy()

    #I think we should futz with timings later - anyway the below is incorrect
    # we don't want to get rid of people with releases dates outside the time period,
    #because they could have commited a crime during the time period.

    # INMT4BB1['release_after_begin_date'] = np.where(
    #     (INMT4BB1['release_date_with_imputation'] > begin_date), 1, 0).copy()
    # INMT4BB1['release_before_end_date'] = np.where(
    #     (INMT4BB1['release_date_with_imputation'] < end_date), 1, 0).copy()
    # INMT4BB1['in_time_window'] = 0
    # INMT4BB1.loc[(INMT4BB1['release_after_begin_date'] > 0) &
    #              (INMT4BB1['release_before_end_date'] > 0), 'in_time_window'] = 1
    #
    # # INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 62, :]
    # # Only keep people with releases in our time window
    # INMT4BB1 = INMT4BB1.groupby(
    #     'INMATE_DOC_NUMBER').filter(lambda x: x['in_time_window'].max() == 1)

    INMT4BB1.tail(10)

    # Number of remaining people
    INMT4BB1['INMATE_DOC_NUMBER'].unique().shape

    return INMT4BB1
################################################################################
                        # MERGE INMT4BB1 AND OFNT3CE1
################################################################################
def merge_offender_inmate_df(OFNT3CE1, INMT4BB1):
    '''
    Merge the inmate and offender pandas dataframes.

    Inputs:
        - OFNT3CE1: offender pandas dataframes
        - INMT4BB1: inmates pandas dataframe
    '''
    # OFNT3CE1.dtypes
    merged = OFNT3CE1.merge(INMT4BB1,
                            left_on=['OFFENDER_NC_DOC_ID_NUMBER',
                                     'COMMITMENT_PREFIX',
                                     'SENTENCE_COMPONENT_NUMBER'],
                            right_on=['INMATE_DOC_NUMBER',
                                      'INMATE_COMMITMENT_PREFIX',
                                      'INMATE_SENTENCE_COMPONENT'],
                            how='outer')
    merged.head()
    # Find the people who have been to jail that we want to keep
    ids_to_keep = merged['INMATE_DOC_NUMBER'].unique()
    ids_to_keep = ids_to_keep[~np.isnan(ids_to_keep)]
    merged = merged.loc[merged['OFFENDER_NC_DOC_ID_NUMBER'].isin(
                list(ids_to_keep)), : ]

    return merged


    # merged.to_csv("merged_subset.csv")

    #Create a clean sentence end date
    # merged.loc[merged[
    #         'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
    #         merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], :].shape
    # merged.loc[merged[
    #         'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
    #         merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], : ][['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX', 'SENTENCE_COMPONENT_NUMBER','SENTENCE_BEGIN_DATE_(FOR_MAX)', 'SENTENCE_EFFECTIVE(BEGIN)_DATE']]

    #Actually I think this should go after collapsing all the counts of a crime into one event.
    #keep only people who were released during our time period FOR A FELONY
    # merged.shape
    # merged['commited_felony_in_time_period'] = np.where(
    #     (merged['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON') &
    #     (merged['release_date_with_imputation'] > begin_date) &
    #     (merged['release_date_with_imputation'] < end_date)
    #     , 1, 0)
    # merged['commited_felony_in_time_period']
    # merged = merged.groupby(
    #     'OFFENDER_NC_DOC_ID_NUMBER').filter(lambda x: x['commited_felony_in_time_period'].max() == 1)

    # merged.shape
    # merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'].isnull().any()
    # merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'].isnull().any()
    # merged.head(10)



def collapse_counts_to_crimes(merged, begin_date):
    '''
    Create a dataframe to put into the ml pipeline.

    The dataframe to put in the pipeline will have 1 row for every crime
    (instead of for every count)

    Inputs:
        - merged: the merged offender and inmate dataframe
        - begin date
        - end date
    '''
    #filter for the counts with release dates before the timeframe
    #we'll filter for crimes with release dates after the timeframe
    time_mask = (merged['release_date_with_imputation'] > begin_date)
    final = merged[time_mask]
    final['release_date_with_imputation'].describe()

    # #drop people that never had a felony during the time period
    # final['crime_felony_or_misd'] = np.where(
    #     (final['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON'), 1, 0).copy()
    # final = final.groupby(
    #     'OFFENDER_NC_DOC_ID_NUMBER').filter(lambda x: x['crime_felony_or_misd'].max() == 1)

    #collapse all counts of a crime into one event
    final['crime_felony_or_misd'] = np.where(final['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0).copy()
    crime_label = final.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX']).apply(lambda x: x['crime_felony_or_misd'].sum()).to_frame().reset_index(
                        ).rename(columns={0: 'num_of_felonies'})

    crime_label['crime_felony_or_misd'] = np.where(crime_label['num_of_felonies'] > 0, 'FELON', 'MISD').copy()

    #assign a begin date and an end date to each crime
    release_date = final.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX']
                                    ).agg({'release_date_with_imputation': 'max',
                                           'SENTENCE_EFFECTIVE(BEGIN)_DATE': 'min'}
                                    ).reset_index()

    #merge together to know if a crime is a misdeamonor or felony
    crime_w_release_date = release_date.merge(crime_label, on=['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'], how='outer')

    crime_w_release_date = crime_w_release_date.sort_values(['OFFENDER_NC_DOC_ID_NUMBER', 'release_date_with_imputation'])
    crime_w_release_date = crime_w_release_date[['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX', 'SENTENCE_EFFECTIVE(BEGIN)_DATE', 'release_date_with_imputation', 'crime_felony_or_misd']]
    crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'])
    return crime_w_release_date


def create_time_since_last_release_df(crime_df):
    '''
    Creates a dataframe unique on OFFENDER_NC_DOC_ID_NUMBER and COMMITMENT_PREFIX (a person and a crime),
    and indicates the time since the person's last felony.

    Helper function for create_df_for_ml_pipeline
    '''
    recidivate_df = pd.DataFrame(columns=['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX', 'time_since_last_release'])
    for offender, crimes in crime_df.groupby('OFFENDER_NC_DOC_ID_NUMBER'):
        crimes = crimes.reset_index()
        last_felony_release_date = pd.to_datetime('0001-01-01', errors = 'coerce')
        for idx, crime in crimes.iterrows():
            if crime['crime_felony_or_misd'] == "FELON" and idx == 0:
                last_felony_release_date = pd.to_datetime(crime['release_date_with_imputation'])
            elif (idx > 0) and (last_felony_release_date):
                time_since_last_release = pd.to_datetime(crime['SENTENCE_EFFECTIVE(BEGIN)_DATE']) - last_felony_release_date
                recidivate_df.loc[len(recidivate_df)] = [crime['OFFENDER_NC_DOC_ID_NUMBER'], crime['COMMITMENT_PREFIX'], time_since_last_release]
                #set the new last crime date to be the current felongy release date
                if crime['crime_felony_or_misd'] == "FELON":
                    last_felony_release_date = pd.to_datetime(crime['release_date_with_imputation'])

    #merge the recidivate_df to the original crime_df
    crime_w_time_since_release = crime_df.merge(recidivate_df, on=['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'])

    return crime_w_time_since_release

def create_time_since_last_release_df_v2(crime_df):
    for index in range(1, crime_df.shape[0]):
      for reverse_index in range(index-1, -1, -1):
        # if the past row is the same person id:
        if crime_df.loc[index, 'OFFENDER_NC_DOC_ID_NUMBER'] == crime_df.loc[reverse_index, 'OFFENDER_NC_DOC_ID_NUMBER']:
          if crime_df.loc[reverse_index, 'crime_felony_or_misd'] == 'FELON':
            crime_df.loc[index, 'time_of_last_felony_release'] = crime_df.loc[reverse_index, 'release_date_with_imputation']
            break
        # if the past row is NOT the same person id, go to the next row
        else:
          break

    return crime_df

def create_recidvate_label(crime_w_release_date, recidviate_definition_in_days):
    crime_w_release_date['recidivate'] = 0
    diff = crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] - crime_w_release_date['time_of_last_felony_release']
    crime_w_release_date.loc[diff < pd.to_timedelta(365, 'D'), 'recidivate'] = 1

    crime_sub =
    crime_w_release_date.head()
    crime_sub['time_of_last_felony_release'] = None
    crime_sub = crime_sub.sort_values(['OFFENDER_NC_DOC_ID_NUMBER','release_date_with_imputation'])
    crime_sub_time_since_release = create_time_since_last_release_df_v2(crime_sub)
crime_sub_time_since_release
    crime_w_release_date.dtypes
    crime_sub





# ADD FEATURES
# create a new data frame that has the total number of incidents with the law
# for each person (guilty or innocent, went to jail or not)
total_number_of_counts = merged.groupby('OFFENDER_NC_DOC_ID_NUMBER').count(
        )['COMMITMENT_PREFIX'].to_frame().reset_index(
        ).rename(columns={'COMMITMENT_PREFIX': 'total_num_counts'})
total_number_of_counts.head()

# create a new data frame that has the total number of incarceral events
total_number_of_incarcerations = merged.groupby(
    ['INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX']).count(
    ).reset_index(
    )[['INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX']].groupby(
            'INMATE_DOC_NUMBER').count().reset_index().rename(
            columns={
            'INMATE_COMMITMENT_PREFIX': 'total_number_of_incarcerations'})

total_number_of_incarcerations.head()
total_number_of_incarcerations.describe()
myhist = total_number_of_incarcerations['total_number_of_incarcerations'].hist()
total_number_of_incarcerations['total_number_of_incarcerations'].quantile(.99)
merged.head()
# did they recidivate within 2 years of last arrest?
#flag earliest time released in the given time period
#time between each release and the release before it
#if there is only one release, then recidivate dummy = 0
#if there is more than one release, and less than 24 months earlier, then recidivate
#dummy is 1.

merged.groupby('INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX')

'release_date_with_imputation'













































first_release = merged.sort_values(
    "clean_ACTUAL_SENTENCE_END_DATE").groupby(
    "OFFENDER_NC_DOC_ID_NUMBER", as_index=False)[
    'clean_ACTUAL_SENTENCE_END_DATE'].first()

merged = merged.merge(first_release,
                      on='OFFENDER_NC_DOC_ID_NUMBER',
                      how='left')
merged.head()

merged = merged.rename(columns={"clean_ACTUAL_SENTENCE_END_DATE_x":
                                "clean_ACTUAL_SENTENCE_END_DATE",
                                "clean_ACTUAL_SENTENCE_END_DATE_y":
                                "first_release_date"})



# Should we use SENTENCE_BEGIN_DATE_(FOR_MAX) or SENTENCE_EFFECTIVE(BEGIN)_DATE
# here?
merged.shape
merged.loc[merged[
        'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
        merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], :].shape
merged.loc[merged[
        'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
        merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], : ][['OFFENDER_NC_DOC_ID_NUMBER','SENTENCE_BEGIN_DATE_(FOR_MAX)', 'SENTENCE_EFFECTIVE(BEGIN)_DATE']]









merged.loc[merged['first_release_date'].isnull()==False, :].head(10)
merged.loc[merged['first_release_date'].isnull()==False, :].groupby('OFFENDER_NC_DOC_ID_NUMBER').count()

merged['time_elapsed'] = merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'] - \
                         merged['first_release_date']
merged['outcome'] = 0
merged.loc[(merged['time_elapsed'] >= '0 days') &
           (merged['time_elapsed'] <= '730 days'), 'outcome'] = 1

merged.loc[merged['first_release_date'].isnull()==False, :].head(10)
merged.head(11)

# Do we want to only keep people with real values for
# clean_ACTUAL_SENTENCE_END_DATE?


# Watch out for cases like this where we have consecutive sentences
merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114, :]

merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114,
                      ['DATE_OFFENSE_COMMITTED_-_BEGIN',
                      'DATE_OFFENSE_COMMITTED_-_END',
                      'OFFENDER_NC_DOC_ID_NUMBER',
                      'SENTENCE_BEGIN_DATE_(FOR_MAX)',
                      'clean_ACTUAL_SENTENCE_END_DATE',
                      'first_release_date',
                      'time_elapsed',
                      'outcome']]


INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 62, :]
OFNT3CE1.loc[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == 62, :]
merged.loc[merged['OFFENDER_NC_DOC_ID_NUMBER'] == 62,
                      ['PROJECTED_RELEASE_DATE_(PRD)',
                      'DATE_OFFENSE_COMMITTED_-_BEGIN',
                      'DATE_OFFENSE_COMMITTED_-_END',
                      'OFFENDER_NC_DOC_ID_NUMBER',
                      'SENTENCE_BEGIN_DATE_(FOR_MAX)',
                      'clean_ACTUAL_SENTENCE_END_DATE',
                      'first_release_date',
                      'time_elapsed',
                      'outcome']]


# NOTE TODO: aggregate variables and collapse to the most recent incarceration
# Eventually we will only want to keep the row with a
# clean_ACTUAL_SENTENCE_END_DATE that matches first_release_date
# That will require spreading outcome within each person.
merged.groupby('OFFENDER_NC_DOC_ID_NUMBER')['outcome'].max()


















OFNT9BE1 = pd.read_csv(
    "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\ncdoc_data\\data\\preprocessed\\OFNT9BE1.csv")
=======
    crime_w_release_date = release_date.merge(crime_label,
                            on=['OFFENDER_NC_DOC_ID_NUMBER',
                                     'COMMITMENT_PREFIX'],
                            how='outer')
    # crime_w_release_date = crime_w_release_date.drop(columns=['num_of_felonies'], axis=1)
    crime_w_release_date.head(5)
    #create a variable of time since last release
    crime_w_release_date['release_date_final'] = pd.to_datetime(crime_w_release_date['crime_release_data_w_imputation'])
    diff_in_dates = np.where(crime_w_release_date['num_of_felonies'] > 1, (crime_w_release_date.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'])['release_date_final'].diff(periods=-1)), 0)
    diff_in_dates
    test = crime_w_release_date.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'])['release_date_final'].diff().reset_index()
    test
    # test.describe()
    # time_since_last_release =
#
#
#
#
# #     diff_in_dates = crime_w_release_date.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'])[].diff(periods=-1, axis=0)
# #     diff_in_dates
# #
# #     #drop
# #
# # crime.head()
# # ADD FEATURES
# # create a new data frame that has the total number of incidents with the law
# # for each person (guilty or innocent, went to jail or not)
# total_number_of_counts = merged.groupby('OFFENDER_NC_DOC_ID_NUMBER').count(
#         )['COMMITMENT_PREFIX'].to_frame().reset_index(
#         ).rename(columns={'COMMITMENT_PREFIX': 'total_num_counts'})
# total_number_of_counts.head()
#
# # create a new data frame that has the total number of incarceral events
# total_number_of_incarcerations = merged.groupby(
#     ['INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX']).count(
#     ).reset_index(
#     )[['INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX']].groupby(
#             'INMATE_DOC_NUMBER').count().reset_index().rename(
#             columns={
#             'INMATE_COMMITMENT_PREFIX': 'total_number_of_incarcerations'})
#
# total_number_of_incarcerations.head()
# total_number_of_incarcerations.describe()
# myhist = total_number_of_incarcerations['total_number_of_incarcerations'].hist()
# total_number_of_incarcerations['total_number_of_incarcerations'].quantile(.99)
# merged.head()
# # did they recidivate within 2 years of last arrest?
# #flag earliest time released in the given time period
# #time between each release and the release before it
# #if there is only one release, then recidivate dummy = 0
# #if there is more than one release, and less than 24 months earlier, then recidivate
# #dummy is 1.
#
# merged.groupby('INMATE_DOC_NUMBER', 'INMATE_COMMITMENT_PREFIX')
#
# 'release_date_with_imputation'
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# first_release = merged.sort_values(
#     "clean_ACTUAL_SENTENCE_END_DATE").groupby(
#     "OFFENDER_NC_DOC_ID_NUMBER", as_index=False)[
#     'clean_ACTUAL_SENTENCE_END_DATE'].first()
#
# merged = merged.merge(first_release,
#                       on='OFFENDER_NC_DOC_ID_NUMBER',
#                       how='left')
# merged.head()
#
# merged = merged.rename(columns={"clean_ACTUAL_SENTENCE_END_DATE_x":
#                                 "clean_ACTUAL_SENTENCE_END_DATE",
#                                 "clean_ACTUAL_SENTENCE_END_DATE_y":
#                                 "first_release_date"})
#
#
#
# # Should we use SENTENCE_BEGIN_DATE_(FOR_MAX) or SENTENCE_EFFECTIVE(BEGIN)_DATE
# # here?
# merged.shape
# merged.loc[merged[
#         'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
#         merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], :].shape
# merged.loc[merged[
#         'SENTENCE_BEGIN_DATE_(FOR_MAX)'] !=
#         merged['SENTENCE_EFFECTIVE(BEGIN)_DATE'], : ][['OFFENDER_NC_DOC_ID_NUMBER','SENTENCE_BEGIN_DATE_(FOR_MAX)', 'SENTENCE_EFFECTIVE(BEGIN)_DATE']]
#
#
#
#
#
#
#
#
#
# merged.loc[merged['first_release_date'].isnull()==False, :].head(10)
# merged.loc[merged['first_release_date'].isnull()==False, :].groupby('OFFENDER_NC_DOC_ID_NUMBER').count()
#
# merged['time_elapsed'] = merged['SENTENCE_BEGIN_DATE_(FOR_MAX)'] - \
#                          merged['first_release_date']
# merged['outcome'] = 0
# merged.loc[(merged['time_elapsed'] >= '0 days') &
#            (merged['time_elapsed'] <= '730 days'), 'outcome'] = 1
#
# merged.loc[merged['first_release_date'].isnull()==False, :].head(10)
# merged.head(11)
#
# # Do we want to only keep people with real values for
# # clean_ACTUAL_SENTENCE_END_DATE?
#
#
# # Watch out for cases like this where we have consecutive sentences
# merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114, :]
#
# merged.loc[ merged['OFFENDER_NC_DOC_ID_NUMBER'] == 114,
#                       ['DATE_OFFENSE_COMMITTED_-_BEGIN',
#                       'DATE_OFFENSE_COMMITTED_-_END',
#                       'OFFENDER_NC_DOC_ID_NUMBER',
#                       'SENTENCE_BEGIN_DATE_(FOR_MAX)',
#                       'clean_ACTUAL_SENTENCE_END_DATE',
#                       'first_release_date',
#                       'time_elapsed',
#                       'outcome']]
#
#
# INMT4BB1.loc[INMT4BB1['INMATE_DOC_NUMBER'] == 62, :]
# OFNT3CE1.loc[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == 62, :]
# merged.loc[merged['OFFENDER_NC_DOC_ID_NUMBER'] == 62,
#                       ['PROJECTED_RELEASE_DATE_(PRD)',
#                       'DATE_OFFENSE_COMMITTED_-_BEGIN',
#                       'DATE_OFFENSE_COMMITTED_-_END',
#                       'OFFENDER_NC_DOC_ID_NUMBER',
#                       'SENTENCE_BEGIN_DATE_(FOR_MAX)',
#                       'clean_ACTUAL_SENTENCE_END_DATE',
#                       'first_release_date',
#                       'time_elapsed',
#                       'outcome']]
#
#
# # NOTE TODO: aggregate variables and collapse to the most recent incarceration
# # Eventually we will only want to keep the row with a
# # clean_ACTUAL_SENTENCE_END_DATE that matches first_release_date
# # That will require spreading outcome within each person.
# merged.groupby('OFFENDER_NC_DOC_ID_NUMBER')['outcome'].max()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# OFNT9BE1 = pd.read_csv(
#     "C:\\Users\\edwar.WJM-SONYLAPTOP\\Desktop\\ncdoc_data\\data\\preprocessed\\OFNT9BE1.csv")
>>>>>>> 9ab3b04a77d551fd2f985eb63561432b84c5298a
