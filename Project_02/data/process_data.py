import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    loading message and category files and saving them into a dataframe
    
    Input:
    messages_filepath filepath to messages csv file
    categories_filepath filepath to categories csv file
    
    Output:
    df dataframe based on merging of messages and categories files
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    
    return df

def clean_data(df):
    '''
    clean_data
    clean the dataframe by extracting categories into separate columns
    converting the columns to integers and dropping duplications in the dataframe
    
    Input:
    df dataframe based on merging of messages and categories files
    
    Output:
    df clean dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories[0:1]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()
    
    # print(category_colnames[0])
    # rename the columns of `categories` [0] is needed because category_colnames is a nested list
    categories.columns = category_colnames[0]
    # print(categories)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    categories.replace(2, 1, inplace=True)
    
    print(categories)
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace = True)
    # print(df.head(5))
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    # print(df.head(5))
    return df

def save_data(df, database_filename):
    '''
    save_data
    saving the dataframe into a sql databank under defined path
    
    Input:
    df clean dataframe
    database_filename path for saving the databank
    
    Output:
    databank under defined path
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('ETL_Pipeline', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        # deleting old database in the directory if there is any
        dir_name = os.getcwd()+"/data/"
        print(dir_name)
        for item in os.listdir(dir_name):
            if item.endswith(".db"):
                os.remove(os.path.join(dir_name, item))

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}\n    OUTPUTFILE: {}'
              .format(messages_filepath, categories_filepath, database_filepath))
        
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python data/process_data.py'\
              'data/disaster_messages.csv data/disaster_categories.csv'\
              'data/ETL_Pipeline.db')


if __name__ == '__main__':
    main()
