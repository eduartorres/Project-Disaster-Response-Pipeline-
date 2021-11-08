# Project-Disaster-Response-Pipeline-
About This Project is part of Data Science Nanodegree Program by Udacity. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

## Introduction

The repo is a project in the Udacity Data Scientist Nanodegree Program.

The dataset (in the data folder) comes from [appen](https://appen.com/) which contains about 26k text messages from news, social media, and some other sources when some disasters happened. The project aim is to classify a disaster text messages into several categories which then can be transmited to the responsible entity.

## Instructions to run the code:
1. Run the following commands in the project's root directory to set up the database and model.

- To run ETL Pipeline that cleans data and stores in database > `python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db`

- To run Machine Learning Pipeline that trains classifier and saves the classifier > `python train_classifier.py ../data/disaster_response_db.db classifier.pkl`

2. Run the following command in the app's directory to run your web app. `python run.py`

3. Go to http://0.0.0.0:3001/ Or Go to http://localhost:3001/

## Requirements
- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

## Files Descriptions
The files structure is arranged as below:

app
- | - templates
- | |- master.html # main page of web app
- | |- go.html # classification result page of web app
- |- run.py # Flask file that runs app

data
- |- disaster_categories.csv # data to process
- |- disaster_messages.csv # data to process
- |- process_data.py
- |- disaster_response_db.db # database to save clean data to
- |- ETL Pipeline Preparation.ipynb

models
- |- train_classifier.py
- |- classifier.pkl # saved model
- |- ML Pipeline Preparation.ipynb

README.md

## Licensing, Authors, Acknowledgements
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

As additional material I used an article published by Jin Tong entitled: ["Learning from Udacity Disaster Response Projects"](https://medium.com/@jtatl/udacity-disaster-response-pipeline-project-learning-be2be43878e6), which was of great help for the development of this project. 
