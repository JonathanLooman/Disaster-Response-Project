# Disaster Response Pipeline Project

## Project motivation 
This project was completed as part of the Udacity Data science nanodegree. This project transformed data and loaded it into a sqlite3 database, then created a model which trained with cross validation in pipelines to aviod data leakage and finally shows results on a webapp. This project demonstrates skills in data engineering, machine learning and software development.

## Requirements
This project should be run with the following python libraries
- numpy 1.18.5
- pandas 1.0.5
- matplotlib 3.2.2
- seaborn 0.10.1
- scipy 1.5.0
- sqlite3 3.32.3
- nltk 3.5
- flask 1.1.2
- joblib 0.16.0
- sqlalchemy 1.3.18
- sklearn 0.23.1
- xgboost 1.1.1

## Files in repository
- app/run.py -> creates webapp to access 
- method_notebook -> demonstrating methodology on ETL and ML process
- app/templets/go.html ->  
- app/templets/master.html ->  
- data/disaster_categories.csv -> classifications of disasters
- data/disaster_database.db -> cleaned data in database
- data/disaster_messages.csv -> disaster messages
- data/process_data.py -> combines and cleans disaster_categories.csv and disaster_messages and loads into disaster_database


- models/firstmodel_XGBoost.sav -> classification model
- train_classifier -> python script used to train firstmodel_XGBoost


## Results
Please find a symmary in a blog post here

