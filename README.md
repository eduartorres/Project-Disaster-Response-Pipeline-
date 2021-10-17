# Project-Disaster-Response-Pipeline-
About This Project is part of Data Science Nanodegree Program by Udacity. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

## Introduction

The repo is a project in the Udacity Data Scientist Nanodegree Program.

The dataset (in the data folder) comes from [appen](https://appen.com/) which contains about 26k text messages from news, social media, and some other sources when some disasters happened.

## Instructions to run the code:
1. Run the following commands in the project's root directory to set up the database and model.

- To run ETL Pipeline that cleans data and stores in database > `<python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db>`

- To run Machine Learning Pipeline that trains classifier and saves the classifier > `<python train_classifier.py ../data/disaster_response_db.db classifier.pkl>`

2. Run the following command in the app's directory to run your web app. `<python run.py>`

3. Go to http://0.0.0.0:3001/

## Dependencies
- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly
