# Disaster Response Pipeline Project

### Summary

A pipeline to extract, clean and store data in a database, and train and deploy a classifier to classify disaster messages.

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files

app
- templates
    - master.html # main page of web app
    - go.html # classification result page of web app
- run.py # Flask file that runs app

data
- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py # file to run and lo
- DisasterResponse.db # database to save clean data to

models
- train_classifier.py # file that trains model
- classifier.pkl # saved model

README.md

### Installation

1. Download this repo and unzip to your favorite location
2. Install dependecies using pip

### Acknowledgements

Thank you udacity for providing input, knowledge and templates. 
