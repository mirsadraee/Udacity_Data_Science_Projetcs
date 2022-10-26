# Disaster Response Pipeline Project
In this project, a model is made to classify messages during disasters. The messages are categorized into 36 categories. The classification of messages helps the related organisations to consider advacne measures. To send any news message during disaster time, it is necessary to develop a tool to classify messages. To do so, two pipelines, i.e. ETL for extraction, transforming and loading data to a databank was developed and connected to a ML pipeline which can classify messages based on their contents. A web app was also developed in this project as user interface for receiving any message and assigning it to specific category.

### Table of Contents
1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [Instructions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>
In this project, an ETL pipeline firstly extracts the data from two files (messages and categories) and merge them into a single databank file. The a ML pipeline reads the databank and tries to classify the messages based on their contents into specific categories. Finally, the model is uploaded to a web application for use and test.

## Installation<a name="installation"></a>
The standard data science libraries of Python are only needed. The code should run with no issues using Python versions 3.

Following files are available in the repository:
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- ETL_Pipeline.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # the model will be saved to a pkl file by following the below instruction. (the file is not available in the repository)
README.md

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