# Functions utilized to clean data Tables
### This notebook is comprised of various functions used to clean and format the EMR data tables

# Februrary 17th, 2025

# Maxime Bouthillier

import pandas as pd
from datetime import datetime

def readmission(dataframe, read_time): 

    '''
    This function will identify all instances of readmission within 
    the user defined timeframe (read_time). 

    This is accomplished by the following:

        The function takes the given dataframe and iterates over all 
        unique subject_ids. If there are multiple entries, a subset dataframe 
        named "mult_entries" is then created and the new 'read_flag' 
        variable = 2 on all entries. If there is a singular entry, the new 
        readmission variable, 'read_flag', = 0. 

        A nested forloop is then utilized to iterate over the "mult_entries" 
        dataframe. An if statement is then imployed to check the time between the 
        previous discharge time to the current admission time. If this time delta is
        less than the user defined read_time, a readmission flag (=1) is appended
        to the previous entry, and a readmission flag of 2 is appeneded to the 
        current entry.

        All readmission flag entries of =2 is then removed from the dataset, as any 
        subject_id entries after the readmission priod is to be removed. Moreover, 
        Any additional subject_id entries with a time delta greater than the time
        delta is also removed (flaged with 2) as to not confuse the model structure
        given the high-dimensionality of the dataset.


    Note: the dataframe is filtered by earliest to the latest date. This ensures the
    funcationality of the nested forloop.
    '''

    # Unsure why but re-importing datetime allows for the use of the time.delta function
    import datetime 

    dataframe = dataframe.sort_values('admittime')    
    subjects = dataframe['subject_id'].unique()
    read_30d = []

    for j in subjects:                                                                                                                         

        if len(dataframe.loc[(dataframe['subject_id'] == j)]) >= 2 :                                                                              

            mult_entries = dataframe.loc[(dataframe['subject_id'] == j)].reset_index()                                                            
            dataframe.loc[(dataframe['subject_id'] == j), 'read_flag'] = 2                                                                            
            dataframe.loc[(dataframe['subject_id'] == j) & (dataframe['admittime'] == mult_entries['admittime'][0]), 'read_flag'] = 0               
        
            for k in range(len(mult_entries['admittime'])-1):
                if abs(mult_entries['dischtime'][k] - mult_entries['admittime'][k+1]) <= datetime.timedelta(days=read_time):                        
                    dataframe.loc[(dataframe['subject_id'] == j) & (dataframe['admittime'] == mult_entries['admittime'][k]) , 'read_flag'] = 1      
                    dataframe.loc[(dataframe['subject_id'] == j) & (dataframe['admittime'] != mult_entries['admittime'][k]) , 'read_flag'] = 2

        else:
            dataframe.loc[(dataframe['subject_id'] == j), 'read_flag'] = 0


    dataframe = dataframe.drop(dataframe[dataframe['read_flag'] == 2].index)  
     
    return(dataframe)


def as_datetime(dataframe, column):

    '''
    Converts the user specified columns' entries into datatime object. 
    This allows for future use of arithmetic operations. This returns the entire dataframe,
    not only the specified column.
    '''

    dataframe[column] = dataframe[column].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return dataframe



def subject_subset(dataframe, adm_df,  column):

    '''
    Subsets the provided dataframe based on the provided datetime column using only subjects identified in the
    adm_df - which needs to be provided. This returns only the entires associated with the correct patient admission.
    Returns the cleaned subsetted dataframe. 
    '''

    dataframe_new = pd.DataFrame()
    subjects = list(adm_df['subject_id'])

    for i in subjects:
        if i in list(dataframe['subject_id']):

            admittime = adm_df.loc[adm_df['subject_id'] ==  i, 'admittime'].iloc[0]
            dischtime = adm_df.loc[adm_df['subject_id'] ==  i, 'dischtime'].iloc[0]

            dataframe_new = pd.concat([dataframe.loc[(dataframe['subject_id'] == i) & (dataframe[column] >= admittime) & 
                (dataframe[column] <= dischtime)], dataframe_new])

    return(dataframe_new)


def subjectcheck(dataframe_1, dataframe_2):

    ''''''


    subjects = list(dataframe_1['subject_id'].unique())
    crosscheck = []

    for i in subjects:
        if i in list(dataframe_2['subject_id']):
            crosscheck.append(i)
    
    return(crosscheck)


def check_nan(dataframe):

    '''Simple functino checking whether a NaN value is present within the given dataset'''

    for i in list(dataframe.columns):
        check_nan = dataframe[i].isnull().values.any()
        if check_nan == True:
            print(i, "NaN value found")
            raise SystemExit("Stopping execution")