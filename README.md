# flask-to-fastapi-ml

A simple machine learning model deployed to Flask and FastAPI

## Description

 - The purpose of this repo is to assess Flask to FastAPI migration
 - Model accepts 11 features and determines loan eligility

## Disclaimer:

 - The classification model used below should not be used in any real world scenario.
 - The model itself is a toy example and is flawed. There is one heavily weighted feature.
 
## Usage
```bash
cd /PATH/TO/FILES

python flask_app.py

or 

python fastapi_app.py
```

## Interacting with Model

You have multiple options:

- Open interact_with_model.ipynb and run the cells
- For FastAPI, navigate to http://127.0.0.1:8000/docs and "Try it out"
- [Try it with Postman](https://www.postman.com/downloads/)
- Command line

 ## Contents
``` 
├── data.csv #  Contains labelled training data 
├── fastapi_app.py #  FastAPI App - Imports * from functions.py
├── flask_app.py #  Flask App - Imports * from functions.py
├── functions.py #  Custom functions and variables.
├── grid_rfc.sav #  Random Forest Classifier
└── interact_with_model.ipynb #  Notebook for interacting with both models
```
