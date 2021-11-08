# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 16:51:36 2021

@author: FELIPE
"""

# Note: Read the header before running
# =============================================================================
# >>> Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree) <<<

# How to execute this file
# Sample Script Syntax:
# > python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite destination db>

# Sample script execution:
# > python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db

# =============================================================================

# ETL PIPELINE PREPARATION

# Loading libraries

import pandas as pd
from sqlalchemy import create_engine
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# ============================================================================
# Module to load messages
def load_mess_categ(dataset_messages, dataset_category):
    """
    >>> This loads messages dataset with categories dataset
    
    >>> Function arguments:
        - dataset_messages: Path to the csv file containing messages
        - dataset_category: Path to the csv file containing categories
    
    >>> Function output:
        df: Merged dataset with the messages and categories datasets
    """
    
    messages_dataset = pd.read_csv(dataset_messages)
    categories_dataset = pd.read_csv(dataset_category)
    df = messages_dataset.merge(categories_dataset, on='id')
    return df 

# ============================================================================

# Module to clean categories data
def clean_data_cat(df):
    """
    >>> This function cleans categories data
    
    >>> Function arguments:
        df: merged dataset with messages and categories
        
    >>> Function outputs:
        df: merged dataset containing messages and categories with categories
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # categories columns name
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    # iterating through the category columns in df to keep only the last character of each string
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df

# ============================================================================

# Module to save data to SQLite Database
def save_data_database(df, database_filepath):
    """
    >>> Save the clean dataset into an sqlite database
    
    >>> Function arguments:
        df: Merged data containing messages and categories with categories cleaned up
        database_filepath: Path to SQLite database
    """
    
    engine = create_engine('sqlite:///'+ database_filepath)
    # Naming the database table
    table_name = database_filepath.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    
# ============================================================================

# Main module that executes the data processing functions above
def main():
    """
    Primary function:
        >>> loads messages and categories data
        >>> cleans up categories Data
        >>> Save data to SQLite database
    """
    
    # Print the system arguments
    # print(sys.argv)
    
    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        # parameters of important variables 
        dataset_messages, dataset_category, database_filepath = sys.argv[1:] 
        
        # =====================================================================

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(dataset_messages, dataset_category))
        
        df = load_mess_categ(dataset_messages, dataset_category)
        
        # =====================================================================

        print('Cleaning up data...')
        df = clean_data_cat(df)
        
        # =====================================================================
        
        print('Saving data to SQLite DB : {}'.format(database_filepath))
        save_data_database(df, database_filepath)
        
        print('Process status: cleaned data has been saved to database successfully!')
        
        # ====================================================================
    
    else:
        print("Please provide the arguments how is described in the: \nSample script execution:\n\
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db")

if __name__ == '__main__':
    main()
    