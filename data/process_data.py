import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
     # load messages dataset
    messages = pd.read_csv(messages_filepath, index_col='id')

    # load categories dataset
    categories = pd.read_csv(categories_filepath, index_col='id')

    # merge datasets
    df = pd.concat([messages, categories], axis=1, join="outer")
    return df

def clean_data(df):
    categories_data = []
    for x in df.categories[2].split(';'):
        categories_data.append(x.split('-')[0])

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    categories.columns = categories_data

    remove_columns = lambda x: x.split('-')[-1]

    for column in categories_data:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(remove_columns)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df.drop('categories', axis=1), categories], axis=1)

    # drop duplicates
    return df[df.message.duplicated(keep='first') == False]

def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('table1', engine, index=False) 


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
