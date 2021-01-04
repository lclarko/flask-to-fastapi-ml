# Description:
# This is a toy ML Flask app
# Model accepts 11 features and determines loan eligility
# This is a POC for porting Flask to FastAPI.
#
# Disclaimer:
# The classification model used below should not
# be used in any real world scenario.
# The purpose of this repo is to assess Flask to FastAPI migration

# Flask Imports

from flask import Flask, request
from flask_restful import Resource, Api

# App Imports

import pickle
import pandas as pd
from functions import *  # Imports everything from functions.py

# Instantiate Flask app
app = Flask(__name__)
api = Api(app)

# Load model
model = pickle.load(open('grid_rfc.sav', 'rb'))


class Predict(Resource):
    def post(self):
        json_data = request.get_json()

        df = pd.DataFrame(json_data.values(),
                          index=json_data.keys()).transpose()

        res = model.predict(df)

        return res.tolist()


# Endpoint to communicate with ML Model
api.add_resource(Predict, '/predict')


@app.route('/test', methods=['GET', 'POST'])
def test():
    """
    This is just to test button functionality
    """
    if request.method == 'GET':
        return('<form action="/test" method="post"><input type="submit" value="Send" /></form>')

    elif request.method == 'POST':
        return "OK this is a post method"
    else:
        return("ok")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
