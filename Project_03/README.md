# Recommandations with IBM
In this project, the interactions of users with articles on the IBM Watson Studio platform was analyzed and newrecommandations were given to them based on their interactions. The user may be new or have interaction history and the platform so it makes the recommandation engine more flexible and reliable on suggesting new articles.


### Table of Contents
1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [Instructions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>
The interactions of users with articles on the IBM Watson Studio platform was analyzed and newrecommandations were given to them based on their interactions. The user may be new or have interaction history and the platform so it makes the recommandation engine more flexible and reliable on suggesting new articles.

## Installation<a name="installation"></a>
The standard data science libraries of Python are only needed. The code should run with no issues using Python versions 3.

Following files are available in the repository:
Recommandations_with_IBM.ipynb # jupyter notebook for data processing
Recommandations_with_IBM.pdf # results of engine recommandations in pdf format
Recommandations_with_IBM.html # results of engine recommandations in html format

data
|- articles_community.csv # data for articles in the community
|- user-item-interactions.csv # data for user and item interactions

README.md

## Instructions<a name="files"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - Recommandations_with_IBM.ipynb

        I. Exploratory Data Analysis
        II. Rank Based Recommendations
        III. User-User Based Collaborative Filtering
        IV. Content Based Recommendations (EXTRA - NOT REQUIRED)
        V. Matrix Factorization


2. Go to 'Recommandations_with_IBM' to see the recommandations results of the simulation


## Licensing<a name="licensing"></a>
The project is made based on Udacity project instructions and motivations.