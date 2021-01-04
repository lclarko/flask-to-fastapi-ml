import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

CAT_FEATS = ['Gender',
             'Married',
             'Dependents',
             'Education',
             'Self_Employed',
             'Property_Area']

NUM_FEATS = ['ApplicantIncome',
             'CoapplicantIncome',
             'LoanAmount',
             'Loan_Amount_Term',
             'Credit_History']

###########################
# DATA PROCESSING FUNCTIONS
###########################


class Transformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self


def outliers(data):
    """
    Removes ApplicantIncome and Loan Amount rows less than or equal to three standard deviations from mean
    """
    data = data[np.abs(data.ApplicantIncome-data.ApplicantIncome.mean()) <= (3*data.ApplicantIncome.std())]
    data = data[np.abs(data.LoanAmount-data.LoanAmount.mean()) <= (3*data.LoanAmount.std())]
    return data


def impute_credit(data):
    """
    Imputes credit history binary value.
    Probability for this random choice is hardcoded and based on the distribution of the population.
    """
    data['Credit_History'] = data['Credit_History'].fillna(pd.Series(np.random.choice([1.0,0.0],
                                                                                      p=[0.842199, 0.157801],
                                                                                      size=len(data))))
    return data


def impute_gender(data):
    """
    Imputes gender binary value. Non-Binary gender was not included in dataset.
    Probability for this random choice is hardcoded and based on the distribution of the population.
    """
    data['Gender'] = data['Gender'].fillna(pd.Series(np.random.choice(['Male','Female'],
                                                                      p=[0.81, 0.19],
                                                                      size=len(data))))
    return data


def impute_marriage(data):
    data['Married'] = data['Married'].fillna(pd.Series(np.random.choice(['Yes','No'], 
                                                                        p=[0.65, 0.35],
                                                                        size=len(data))))
    data['Dependents'] = data['Dependents'].replace('3+', 3)
    data['Dependents'] = data[['Dependents']].fillna(0).astype('int16')
    return data


def impute_employment(data):
    data['Self_Employed'] = data['Self_Employed'].fillna(pd.Series(np.random.choice(['Yes','No'],
                                                                                    p=[0.86, 0.14],
                                                                                    size=len(data))))
    return data


def binarizer(data):
    """
    Manually label encodes binary features
    """
    data['Male'] = np.where(data['Gender'] == 'Male', 1, 0)
    data['Graduated'] = np.where(data['Education'] == 'Graduate', 1, 0)
    data['Married'] = np.where(data['Married'] == 'Yes', 1, 0)
    data['Self_Employed'] = np.where(data['Self_Employed'] == 'Yes', 1, 0)
    return data


def dummy(data):
    data = pd.get_dummies(data, columns=['Property_Area'])
    return data


def dummies(data):
    """
    Manually creates labels for property type. sklearn label encoder was breaking the model shape.
    """
    data['Property_Area'] = data['Property_Area'].replace('Rural', 0, regex=True)
    data['Property_Area'] = data['Property_Area'].replace('Semiurban', 1, regex=True)
    data['Property_Area'] = data['Property_Area'].replace('Urban', 2, regex=True)
    return data


def shed(data):
    """
    Drops columns that have been encoded
    """
    data = data.drop(['Gender','Education','Married','Self_Employed'],axis=1)
    return data


def impute_loan_term(data):
    """
    Imputes loan term value with the mean loan term.
    """
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean())
    data['Loan_Amount_Term'] = data['Loan_Amount_Term']/12
    return data

#######################
# PIPELINE
#######################
#
# Numerical and categorical data will have separate pipelines.
# Both pieplines feed into the prediction pipeline.


numeric_transformer = Pipeline(steps=[
    ('credit', Transformer(impute_credit)),
    ('term', Transformer(impute_loan_term)),
    ('imputer', SimpleImputer(strategy='mean', fill_value='missing')),
    ('scaler', StandardScaler())
    ])

categorical_transformer = Pipeline(steps=[
    ('gender', Transformer(impute_gender)),
    ('marriage', Transformer(impute_marriage)),
    ('employment', Transformer(impute_employment)),
    ('dummies', Transformer(dummies)),
    ('shed', Transformer(shed)),
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUM_FEATS),
        ('cat', categorical_transformer, CAT_FEATS)])

rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('rf_clf', RandomForestClassifier())])