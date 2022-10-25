# Disaster Response Pipeline Project

## Project Description


## Installation
The standard data science libraries of Python are only needed. The code should run with no issues using Python versions 3.


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/ETL_Pipeline.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/ETL_Pipeline.db models/ML_Classifier.pkl`

2. Go to `app` directory: `cd app` and run the following command in the app's directory to run your web app.
     `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing
The model are made based on Udacity project instructions.