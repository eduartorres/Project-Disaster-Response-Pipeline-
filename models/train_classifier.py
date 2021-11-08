# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:45:21 2021

@author: FELIPE
"""

# Note: Read the header before running
# =============================================================================
# >>> Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree) <<<

# How to execute this file
# Sample Script Syntax:
# > python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

# Sample Script Execution:
# > python train_classifier.py ../data/disaster_response_db.db classifier.pkl

# =============================================================================

# MACHINE LEARNING PIPELINE

# Loading libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import os
import re
import sys
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy import stats

from scipy.stats import gmean

# import relevant functions/modules from the sklearn
from nltk.corpus import stopwords 
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

# ============================================================================
# Module to load dataset
def load_data_db(database_filepath):
    """
    >>> Load dataset from the database
    
    >>> Function arguments:
        database_filepath: Path to SQLite database
        
    >>> Function output:
        X: dataframe containing features
        Y: dataframe containing labels
        category_names: List of categories name
    """
    # Load data from the database
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    # Dropping 'child_alone' column
    df = df.drop(['child_alone'],axis=1)
    
    # Replacing 2 with 0 to consider it a valid response
    df["related"].replace({2: 0}, inplace=True)
    
    # Selecting variables
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    # Listing the columns
    category_names = y.columns
    
    return X, y, category_names

# ============================================================================

# Module to tokenize text data
def tokenize(text, urlplaceholder="urlplaceholder"):
    """
    >>> Function splits text into words and return the root form of the words
    
    >>> Fucntion arguments:
      text: text messages
      
    >>> Function output:
      root_words: a list of the root form of the message words
    """
    # 1. Replace all urls with a url place holder string
    url_rgx = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # 2. Extract all the urls from text 
    urls = re.findall(url_rgx, text)
    
    # 3. Replace url with a url placeholder string
    for url in urls:
        text = text.replace(url, urlplaceholder)

    # 4. Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # 5. Lemmatization
    lemmat = nltk.WordNetLemmatizer()

    # 6. List of clean tokens
    root_words = [lemmat.lemmatize(w).lower().strip() for w in tokens]
    return root_words

# ============================================================================

# Module to build machine learning model
def build_model():
    """
    >>> Function that builds ML model
    
    >>> Function output:
        A Machine Learning pipeline that process text messages based on AdaBoostClassifier
        
    """
    # Build a pipeline
    pipeline_ada = Pipeline([
        ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
        ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf_transformer', TfidfTransformer())]))
        ])),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))])
    
    # Create Grid search parameters
    param_ada = {'classifier__estimator__learning_rate': [0.01, 0.03, 0.05],
                 'classifier__estimator__n_estimators': [10, 30, 50]}
    
    cv_ada = GridSearchCV(pipeline_ada, param_grid=param_ada, scoring='f1_micro', n_jobs=-1)
    
    return cv_ada

# ============================================================================

# Module to evaluate the built model
def evaluate_model(model, X_test, Y_test, category_names):
    """
    >>> prints out the model performance
    
    >>> Function arguments:
        model: GridSearchCV object
        X_test: Test features
        Y_test: Test labels
        category_names: label names
    """
    Y_pred = model.predict(X_test)
    
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('Average overall accuracy {0:.3f}%'.format(overall_accuracy*100))

    # Print the classification report
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model performance with category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))

# ============================================================================

# Module to save the model as a pickle file
def save_model(model, pickle_filepath):
    """
    >>> This function saves trained model as Pickle file
    
    >>> Function arguments:
        - model: GridSearchCV object
        - pickle_filepath: destination path to save Pickle file
    
    """
    pickle.dump(model, open(pickle_filepath, 'wb'))

# ============================================================================

# Main module that executes the data processing functions above
def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        >>> Extract data from SQLite database
        >>> Train Machine Learning model on training set
        >>> Evaluate model performance
        >>> Save trained model as Pickle file
    
    """
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, Y, category_names = load_data_db(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # ====================================================================
        
        print('Building the model ...')
        model = build_model()
        
        # ====================================================================
        
        print('Training the model ...')
        model.fit(X_train, Y_train)
        
        # ====================================================================
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        # ====================================================================

        print('Saving model to {} ...'.format(pickle_filepath))
        save_model(model, pickle_filepath)
        
        # ====================================================================

        print('Process status: trained model saved successfully!')
        
        # ====================================================================

    else:
         print("Please provide the arguments correctly: \nSample Script Execution:\n\
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl")

if __name__ == '__main__':
    main()