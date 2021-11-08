# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 19:16:27 2021

@author: FELIPE
"""
# Note: Read the header before running
# =============================================================================
# >>> Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree) <<<

# Sample script execution:
# > python run.py

# =============================================================================

# FLASK WEB APP

# Loading libraries

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
import nltk
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# ============================================================================

# loading data
engine = create_engine('sqlite:///../data/disaster_response_db.db')
df = pd.read_sql_table('disaster_response_db_table', engine)

# ============================================================================

# load model
model = joblib.load("../models/classifier.pkl")

# ============================================================================

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # visuals for the genre 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values
    
    
    # create visuals
    # genre graph and category graph 
    graphs = [
            # Graph - genre graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker = dict(
                            color = 'rgb(60,179,113)')
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
            # Graph - category graph    
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean,
                    marker = dict(
                            color = 'rgb(218,165,32)')
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# ============================================================================

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()