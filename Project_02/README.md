# Disaster Response Pipeline Project
The second project in the framework of Udacity Data Scientist Nano Degree course.

### Table of Contents
1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [Instructions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>
In this project, an ETL pipeline firstly extracts the data from two files (messages and categories) and merge them into a single databank file. The a ML pipeline reads the databank and tries to classify the messages based on their contents into specific categories. Finally, the model is uploaded to a web application for use and test.

## Installation<a name="installation"></a>
The standard data science libraries of Python are only needed. The code should run with no issues using Python versions 3.

## Instructions<a name="files"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/ETL_Pipeline.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/ETL_Pipeline.db models/ML_Classifier.pkl`

2. Go to `app` directory: `cd app` and run the following command in the app's directory to run your web app.
     `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing<a name="licensing"></a>
The project is made based on Udacity project instructions.