import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk


'''
function objective: loads in the two datasets and merges them on common id.
input: messages_filepath - route to messages csv.
       categories_filepath - route to categories csv.
output: df - the merged dataframe. 
'''

def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on = 'id')
    
    return df;

'''
function objective: expand out categorical phases, remove grammar and remove duplicates. 
input: df - dataframe with 'catagories' to be split out.
output: df - the processed dataframe.
'''

def clean_data(df):
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = [x[:-2] for x in categories.iloc[0].values];
    categories.columns = category_colnames;
    categories = categories.apply(lambda x: x.str.replace(r"([a-z]+)(-|_)", ""));
    categories = categories.astype('int')
    
    df = df.drop('categories', axis=1);
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(keep='last');

    return df;

'''
function objective: creates an sql engine and saves the cleaned dataset to it.
input: df - the dataframe you want to save.
       database_filename - your engine
output: no returned output, but the cleaned dataframe will be saved in your engine. 
'''

def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///' + database_filename)
#     df.to_sql('disaster_data_cleaned', engine, index=False, if_exists='replace')
    df.to_sql('disaster_messages_tbl', engine, index=False, if_exists='replace')

'''
function objective: Directs the data preparation while providing updates. 
'''

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
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
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()