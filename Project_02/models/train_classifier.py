# import libraries
import sys
import re
import os 
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine

import pickle

def load_data(database_filepath):
    '''
    load_data
    loading databk into a dataframe
    
    Input:
    database_filepath filepath to sql databank file
    
    Output:
    X messages in the databank
    Y categories in the databank
    category_names of the categories in the databank
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('ETL_Pipeline', engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    df.head()
    return X, Y, category_names


def tokenize(text):
    '''
    tokenize
    cleaning the text and removing useless information from it
    
    Input:
    text text messages in the databank
    
    Output:
    clean_tokens clean text for machine learning
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    build_model
    building a pipeline for building a classifier model
        
    Output:
    model classifier models considering the grid search
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [2],
    'clf__estimator__min_samples_split': [2],
    }

    # create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters, cv =3)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    evaludating the model using test dataset
        
    Input:
    model classifier model
    X_test input from test dataset
    Y_test output from test dataset
    
    Output:
    Classification report
    '''
    Y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, Y_pred)

    # for col in Y_test.columns:
    #     print("category: ", col)
    #     classification_report(Y_test[col], Y_pred[col], category_names)
    print(class_report)

def save_model(model, model_filepath):
    '''
    save_model 
    save the model into a pickel file
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        # deleting old pickel file in the directory if there is any
        dir_name = os.getcwd()+"/models/"
        # print(dir_name)
        for item in os.listdir(dir_name):
            if item.endswith(".pkl"):
                os.remove(os.path.join(dir_name, item))

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(category_names)        

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\n'\
              'Example: python models/train_classifier.py'\
              'data/ETL_Pipeline.db models/ML_Classifier.pkl')

if __name__ == '__main__':
    main()
