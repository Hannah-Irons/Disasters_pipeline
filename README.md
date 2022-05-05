# Disaster Response Pipeline Project


## Project Overview

The task of this project was to create an ETL pipeline that takes in disaster messages and categories; cleans and tokenizes it so that it can be used to model and classify a category that a given message would correspond to i.e. flooding, medical aid etc.
 
## Installation 

All libraries are available in Anaconda distribution of Python. The used libraries are:

- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3




### Instructions from Udacity:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the web address in teh terminal to open the homepage
