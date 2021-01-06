# Description:
# This is a migrated toy ML Flask app to FastAPI
# Model accepts 11 features and determines loan eligility
# This is a POC for porting Flask to FastAPI.
#
# Disclaimer:
# The classification model used below should not
# be used in any real world scenario.
# The purpose of this repo is to assess Flask to FastAPI migration

# FastAPI imports

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# App imports

import pickle
import pandas as pd
from functions import *  # Imports everything from functions.py


# Create data model for client request body

class Applicant(BaseModel):
    Gender: Optional[str] = None
    Married: Optional[str] = None
    Dependents: Optional[str] = None
    Education: Optional[str] = None
    Self_Employed: Optional[str] = None
    ApplicantIncome: int
    CoapplicantIncome: Optional[int] = None
    LoanAmount: int
    Loan_Amount_Term: int
    Credit_History: Optional[int] = None
    Property_Area: str


# Instantiate FastAPI app
app = FastAPI()

# Load model
model_pipeline = pickle.load(open('grid_rfc.sav', 'rb'))


def get_assessment(json):
    data = json.dict()
    x = pd.DataFrame(data.values(),
                     index=data.keys()).transpose()
    y = model_pipeline.predict(x)
    return {'Loan Eligibility': int(y)}


# Routes (Endpoints)
@app.post("/predict")
async def predict(json: Applicant):
    pred = get_assessment(json)
    return pred


@app.get('/heartbeat')
async def read_root():
    return {"Hello": "World"}


if __name__ == '__main__':
    uvicorn.run(app)
