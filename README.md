# NLP-Modelling-in-Python

## Introduction
This project is based on using NLP techniques to appropriately classify disaster data provided by Figure Eight Inc (as part of their partnership with udacity Data Science Nanodegree) into categories which can then be used to send messages to an appropriate disaster relief agency.

The output will be a web app which will intake in any input from an emergency worker and get classification results in various categories.This web app will also display visualisations of the data.

## Dependencies:
- A sql lite database
- sqlalchemy
- Flask 
- Plotly

## How to Run this App:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
