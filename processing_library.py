'''
Processing script
Vedika Ahuja, Bhargavi Ganesh, and Pete Rodrigue

This script contains the functions for cleaning and merging for the offender, inmate, and demographic files.
In the schema, this corresponds to tables OFNT3CE1, INMT4BB1, and OFNT3AA1.
The script also contains functions to collapse the counts to crimes, create the recidivate label,
and create the following features: age at first incarceration, number of previous incarcerations.
At the end we also provide a function that collapses counts to crimes and
merges dummy variables onto the master dataframe.
'''
import pandas as pd
import numpy as np
import datetime
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import os

def make_final_df(offender_filepath, inmate_filepath, demographics_filepath, begin_date, end_date, recidivate_definition_in_days):
    '''
    This function loads in, merges, and cleans data to make the final dataframe which is used for the ml pipeline

    Inputs:
        offender_filepath: filepath for offender table
        inmate_filepath: filepath for inmate table
        demographics_filepath: filepath for demographic variables
        begin_date: begin date of timeframe
        end_date: end date of timeframe
        recidivate_definition_in_days: definition of recidivism in days
    Returns:
        final_df
    '''
    OFNT3CE1 = clean_offender_data(offender_filepath)
    print('\tOFNT3CE1 data cleaned and loaded\t', datetime.now())
    INMT4BB1 = clean_inmate_data(inmate_filepath, begin_date, end_date)
    print('\tINMT4BB1 data cleaned and loaded\t', datetime.now())
    merged = merge_offender_inmate_df(OFNT3CE1, INMT4BB1)
    print('\tOFNT3CE1 and INMT4BB1 merged\t', datetime.now())
    crime_w_release_date = collapse_counts_to_crimes(merged, begin_date)
    print("\t collapsed counts to crimes done\t", datetime.now())
    #Subset data to the timeframe of interest
    df_to_ml_pipeline = crime_w_release_date.loc[crime_w_release_date['release_date_with_imputation'] > pd.to_datetime(begin_date)]
    df_to_ml_pipeline = crime_w_release_date.loc[crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] < pd.to_datetime(end_date)]
    df_to_ml_pipeline = df_to_ml_pipeline.reset_index()
    #Add recidivate label
    crimes_w_time_since_release_date = create_time_to_next_incarceration_df(df_to_ml_pipeline)
    crimes_w_recidviate_label = create_recidvate_label(crimes_w_time_since_release_date, recidivate_definition_in_days)
    print('\trecidivate label created\t', datetime.now())
    OFNT3AA1 = load_demographic_data(demographics_filepath)
    crimes_w_demographic = crimes_w_recidviate_label.merge(OFNT3AA1,
                            on='OFFENDER_NC_DOC_ID_NUMBER',
                            how='left')
    print('\tdemographic data loaded and merged on\t', datetime.now())
    #Add age and years in prison features
    crimes_w_demographic['age_at_crime'] = (crimes_w_demographic['SENTENCE_EFFECTIVE(BEGIN)_DATE'] -
                                    pd.to_datetime(crimes_w_demographic['OFFENDER_BIRTH_DATE'])) / np.timedelta64(365, 'D')
    crimes_w_demographic['years_in_prison'] = (crimes_w_demographic['release_date_with_imputation'] - \
                pd.to_datetime(crimes_w_demographic['SENTENCE_EFFECTIVE(BEGIN)_DATE'])) / np.timedelta64(365, 'D')
    crimes_w_demographic = df_w_age_at_first_incarceration(crimes_w_demographic)
    #Add features for number of previous incarcerations
    crimes_w_demographic = create_number_prev_incarcerations(crimes_w_demographic)
    print('\ttime variables added\t', datetime.now())
    final_df = crimes_w_demographic.loc[crimes_w_demographic['crime_felony_or_misd']=='FELON',]

    return final_df

def clean_offender_data(offender_filepath):
    '''
    Takes the offender dataset (OFNT3CE1), cleans it, and outputs a pandas dataframe.

    Inputs:
        offender_filepath: filepath for offender dataset within ncdoc_data folder

    Returns:
        pandas dataframe of cleaned offender file
    '''
    #Read in offender file
    OFNT3CE1 = pd.read_csv(offender_filepath,
        dtype={'OFFENDER_NC_DOC_ID_NUMBER': str,
               'MAXIMUM_SENTENCE_LENGTH': str,
               'SPLIT_SENTENCE_ACTIVE_TERM': str,
               'SENTENCE_TYPE_CODE.5': str,
               'PRIOR_P&P_COMMNT/COMPONENT_ID': str,
               'ORIGINAL_SENTENCE_AUDIT_CODE': str})
    # Create a variable that indicates felony offenses
    OFNT3CE1['is_felony'] = np.where(
        OFNT3CE1['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0)
    doc_ids_with_felony = OFNT3CE1.groupby(
        'OFFENDER_NC_DOC_ID_NUMBER').filter(
                lambda x: x['is_felony'].max() == 1).reset_index(
                )['OFFENDER_NC_DOC_ID_NUMBER'].unique().tolist()
    #Select only rows that are felonies
    OFNT3CE1 = OFNT3CE1[OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].isin(doc_ids_with_felony)]
    #Clean the dates
    OFNT3CE1['clean_SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(
            OFNT3CE1['SENTENCE_EFFECTIVE(BEGIN)_DATE'], errors='coerce')
    #Drop features we don't want to use, because mostly null values
    OFNT3CE1 = OFNT3CE1.drop(['NC_GENERAL_STATUTE_NUMBER',
                              'LENGTH_OF_SUPERVISION',
                              'SUPERVISION_TERM_EXTENSION',
                              'SUPERVISION_TO_FOLLOW_INCAR.',
                              'G.S._MAXIMUM_SENTENCE_ALLOWED',
                              'ICC_JAIL_CREDITS_(IN_DAYS)'], axis=1)
    #Making one person's id a number so we can make them all numeric
    OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'].loc[
            OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] == 'T153879'] = "-999"
    OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'] = pd.to_numeric(
            OFNT3CE1['OFFENDER_NC_DOC_ID_NUMBER'])

    return OFNT3CE1

################################################################################
                        # CLEAN AND READ  INMT4BB1
################################################################################
def clean_inmate_data(inmate_filepath, begin_date, end_date):
    '''
    Takes the inmate dataset (INMT4BB1), cleans it, and outputs a pandas dataframe.
    Where there is no actual sentence end date, we impute the sentence end date
    using projected release date.

    Inputs:
        inmate_filepath: csv file path
        begin_date: The beginning date of the time period of interest - this is the release date we want to look at
        end_date: The end date of the time period

    Returns:
        pandas dataframe of cleaned inmate file
    '''
    INMT4BB1 = pd.read_csv(inmate_filepath)
    #Dropping features we don't want to use, because mostly null values
    INMT4BB1 = INMT4BB1.drop(['INMATE_COMPUTATION_STATUS_FLAG',
                              'PAROLE_DISCHARGE_DATE',
                              'PAROLE_SUPERVISION_BEGIN_DATE'], axis=1)
    #Clean dates, impute projected sentence end date
    INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'] = pd.to_datetime(
                INMT4BB1['ACTUAL_SENTENCE_END_DATE'], errors='coerce')
    INMT4BB1['clean_projected_release_date'] = pd.to_datetime(
                INMT4BB1['PROJECTED_RELEASE_DATE_(PRD)'], errors='coerce')
    INMT4BB1['clean_SENTENCE_BEGIN_DATE_(FOR_MAX)'] = pd.to_datetime(
            INMT4BB1['SENTENCE_BEGIN_DATE_(FOR_MAX)'], errors='coerce')
    INMT4BB1['release_date_with_imputation'] = INMT4BB1[
                                'clean_ACTUAL_SENTENCE_END_DATE']
    INMT4BB1['release_date_with_imputation'] = np.where(
        (INMT4BB1['release_date_with_imputation'].isnull()),
        INMT4BB1['clean_projected_release_date'],
        INMT4BB1['release_date_with_imputation']).copy()
    INMT4BB1['imputed_release_date_flag'] = np.where(
            INMT4BB1['clean_projected_release_date'].notnull() &
            INMT4BB1['clean_ACTUAL_SENTENCE_END_DATE'].isnull(), 1, 0).copy()

    return INMT4BB1


def load_demographic_data(demographics_filepath):
    '''
    Loads and cleans the demographic dataset

    Inputs:
        demographics_filepath: filepath for demographic data

    Returns:
        pandas dataframe of cleaned demographic data
    '''
    #Read in demographic data
    OFNT3AA1 = pd.read_csv(demographics_filepath, dtype={
        'OFFENDER_BIRTH_DATE': str,
        'OFFENDER_GENDER_CODE': str,
        'OFFENDER_RACE_CODE': str,
        'OFFENDER_SKIN_COMPLEXION_CODE': str,
        'OFFENDER_HAIR_COLOR_CODE': str,
        'OFFENDER_EYE_COLOR_CODE': str,
        'OFFENDER_BODY_BUILD_CODE': str,
        'CITY_WHERE_OFFENDER_BORN': str,
        'NC_COUNTY_WHERE_OFFENDER_BORN': str,
        'STATE_WHERE_OFFENDER_BORN': str,
        'COUNTRY_WHERE_OFFENDER_BORN': str,
        'OFFENDER_CITIZENSHIP_CODE': str,
        'OFFENDER_ETHNIC_CODE': str,
        'OFFENDER_PRIMARY_LANGUAGE_CODE': str})
    #Drop variables we don't want to use, because mostly null values
    OFNT3AA1 = OFNT3AA1.drop(['OFFENDER_SHIRT_SIZE', 'OFFENDER_PANTS_SIZE',
                   'OFFENDER_JACKET_SIZE', 'OFFENDER_SHOE_SIZE',
                   'OFFENDER_DRESS_SIZE', 'NEXT_PHOTO_YEAR',
                   'DATE_OF_LAST_UPDATE', 'TIME_OF_LAST_UPDATE'], axis=1)
    #Change variable types
    OFNT3AA1['OFFENDER_HEIGHT_(IN_INCHES)'] = pd.to_numeric(
            OFNT3AA1['OFFENDER_HEIGHT_(IN_INCHES)'])
    OFNT3AA1['OFFENDER_WEIGHT_(IN_LBS)'] = pd.to_numeric(
            OFNT3AA1['OFFENDER_WEIGHT_(IN_LBS)'])
    OFNT3AA1['OFFENDER_NC_DOC_ID_NUMBER'] = OFNT3AA1[
                'OFFENDER_NC_DOC_ID_NUMBER'].astype(str)
    OFNT3AA1['OFFENDER_NC_DOC_ID_NUMBER'] = OFNT3AA1[
                    'OFFENDER_NC_DOC_ID_NUMBER'].str.replace(
                    'T', '', regex=False)
    OFNT3AA1['OFFENDER_NC_DOC_ID_NUMBER'] = pd.to_numeric(
            OFNT3AA1['OFFENDER_NC_DOC_ID_NUMBER'])

    return OFNT3AA1


def merge_offender_inmate_df(OFNT3CE1, INMT4BB1):
    '''
    Merges the inmate and offender pandas dataframes.

    Inputs:
        OFNT3CE1: offender pandas dataframes
        INMT4BB1: inmates pandas dataframe
    Returns:
        merged inmate and offender pandas dataframes
    '''
    merged = OFNT3CE1.merge(INMT4BB1,
                            left_on=['OFFENDER_NC_DOC_ID_NUMBER',
                                     'COMMITMENT_PREFIX',
                                     'SENTENCE_COMPONENT_NUMBER'],
                            right_on=['INMATE_DOC_NUMBER',
                                      'INMATE_COMMITMENT_PREFIX',
                                      'INMATE_SENTENCE_COMPONENT'],
                            how='right')

    return merged


def collapse_counts_to_crimes(merged, begin_date):
    '''
    Reshapes dataframe so that there is 1 row for every crime
    (instead of 1 for every count)

    Inputs:
        merged: merged offender and inmate dataframe
        begin date: beginning date of the time period of interest

    Returns:
        reshaped pandas dataframe
    '''
    #Filter for crimes with release dates after the timeframe begin date
    time_mask = (merged['release_date_with_imputation'] > begin_date)
    final = merged[time_mask]
    #Collapse all counts of a crime into one event
    final['crime_felony_or_misd'] = np.where(final['PRIMARY_FELONY/MISDEMEANOR_CD.'] == 'FELON', 1, 0).copy()
    crime_label = final.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX']).apply(lambda x: x['crime_felony_or_misd'].sum()).to_frame().reset_index(
                        ).rename(columns={0: 'num_of_felonies'})

    crime_label['crime_felony_or_misd'] = np.where(crime_label['num_of_felonies'] > 0, 'FELON', 'MISD').copy()
    #Assign a begin date and an end date to each crime
    release_date = final.groupby(['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX']
                                    ).agg({'release_date_with_imputation': 'max',
                                           'SENTENCE_EFFECTIVE(BEGIN)_DATE': 'min',
                                           'SENTENCE_COMPONENT_NUMBER' : 'count'}
                                    ).reset_index().rename(columns={'SENTENCE_COMPONENT_NUMBER': 'total_counts_for_crime'})
    #Merge collapsed data with released date to know if a crime is a misdeamonor or felony
    crime_w_release_date = release_date.merge(crime_label, on=['OFFENDER_NC_DOC_ID_NUMBER', 'COMMITMENT_PREFIX'], how='outer')
    crime_w_release_date = crime_w_release_date.rename(columns={'num_of_felonies' : 'felony_counts_for_crime'})
    crime_w_release_date = crime_w_release_date.sort_values(['OFFENDER_NC_DOC_ID_NUMBER', 'release_date_with_imputation'])
    crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'] = pd.to_datetime(crime_w_release_date['SENTENCE_EFFECTIVE(BEGIN)_DATE'])

    return crime_w_release_date


def create_time_to_next_incarceration_df(df):
    '''
    Creates a dataframe unique on OFFENDER_NC_DOC_ID_NUMBER and COMMITMENT_PREFIX
    (a person and a crime), and indicates the time since the person's last felony.

    Inputs:
        df: dataframe of crimes with release date

    Returns:
        pandas dataframe with variable indicating time till next incarceration
    '''
    df.sort_values(['OFFENDER_NC_DOC_ID_NUMBER', 'SENTENCE_EFFECTIVE(BEGIN)_DATE'])
    #Arbitrarily large date
    df['start_time_of_next_incarceration'] = datetime.strptime('2080-01-01', '%Y-%m-%d')
    for index in range(0, df.shape[0] - 1):
        if df.loc[index, 'crime_felony_or_misd'] != 'FELON':
            continue
        else:
            if df.loc[index, 'OFFENDER_NC_DOC_ID_NUMBER'] == df.loc[index + 1, 'OFFENDER_NC_DOC_ID_NUMBER']:
                df.loc[index, 'start_time_of_next_incarceration'] = df.loc[index + 1, 'SENTENCE_EFFECTIVE(BEGIN)_DATE']

    return df


def create_recidvate_label(crime_w_release_date, recidviate_definition_in_days):
    '''
    Takes dataframe and creates label for whether or not someone recidivates
    (returns to prison for either a misdemeanor or felony within a specified timeframe)

    Inputs:
        pandas dataframe collapsed by crime
        recidviate_definition_in_days: measure of recidivism (number of days)

    Returns:
        pandas dataframe with a label for whether someone recidivates or not
    '''
    crime_w_release_date['recidivate'] = 0
    diff = (crime_w_release_date['start_time_of_next_incarceration'] -
            crime_w_release_date['release_date_with_imputation'])
    crime_w_release_date.loc[diff < pd.to_timedelta(recidviate_definition_in_days, 'D'), 'recidivate'] = 1

    return crime_w_release_date


################################################################################
# ADD FEATURES: age at first incarceration, number of previous incarcerations
###############################################################################
def df_w_age_at_first_incarceration(input_df):
    '''
    Creates feature of the age of the person at the first time they are arrested or incarcerated

    Input:
        pandas dataframe collapsed by crime

    Returns:
        pandas dataframe with a variable indicating age of first incarceration
    '''
    df = input_df.sort_values(['OFFENDER_NC_DOC_ID_NUMBER', 'SENTENCE_EFFECTIVE(BEGIN)_DATE']).copy()
    df['age_at_first_incarceration'] = (df['SENTENCE_EFFECTIVE(BEGIN)_DATE'] - pd.to_datetime(df['OFFENDER_BIRTH_DATE'])) / np.timedelta64(365, 'D')

    df_grouped = df.groupby(['OFFENDER_NC_DOC_ID_NUMBER']
                                    ).agg({'age_at_first_incarceration' : 'min'}
                                    ).reset_index()
    return input_df.merge(df_grouped, on='OFFENDER_NC_DOC_ID_NUMBER', how='left')


def create_number_prev_incarcerations(df):
    '''
    Creates feature for number of previous incarcerations

    Input:
        pandas dataframe collapsed by crime

    Returns:
        pandas dataframe with a variable indicating number of previous incarcerations
    '''
    df = df.sort_values(['OFFENDER_NC_DOC_ID_NUMBER',
                         'SENTENCE_EFFECTIVE(BEGIN)_DATE'])
    df['number_of_previous_incarcerations'] = 0
    nrows_df = df.shape[0]
    num_previous_incar = 0
    for i in range(1, nrows_df):
        if (df['OFFENDER_NC_DOC_ID_NUMBER'][i] ==
           df['OFFENDER_NC_DOC_ID_NUMBER'][i - 1]):
            num_previous_incar += 1
            df.loc[i, 'number_of_previous_incarcerations'] = num_previous_incar
        else:
            num_previous_incar = 0

    return df

################################################################################
# Functions to make dummies, specific to this dataset, used in ml_functions_library
###############################################################################
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
        to_add = make_count_vars_to_merge_onto_master_df(subset_df, var)
        df = df.merge(to_add, on=['OFFENDER_NC_DOC_ID_NUMBER',
                                              'COMMITMENT_PREFIX'], how='left')
    return df
