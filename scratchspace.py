import pandas as pd
import numpy as np
import matplotlib as plt
import datetime
import os
pd.options.display.max_columns = 100


os.chdir('C:/Users/edwar.WJM-SONYLAPTOP/Desktop/ncdoc_data')

demographics_filepath = "data/preprocessed/OFNT3AA1.csv"


def load_demographic_data(demographics_filepath):
    '''Loads and cleans the demographic dataset'''
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
    OFNT3AA1 = OFNT3AA1.drop(['OFFENDER_SHIRT_SIZE', 'OFFENDER_PANTS_SIZE',
                   'OFFENDER_JACKET_SIZE', 'OFFENDER_SHOE_SIZE',
                   'OFFENDER_DRESS_SIZE', 'NEXT_PHOTO_YEAR',
                   'DATE_OF_LAST_UPDATE', 'TIME_OF_LAST_UPDATE'], axis=1)

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

OFNT3AA1.head()
