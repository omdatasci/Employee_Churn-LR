import pandas as pd
import numpy as np
import pickle

from flask import Flask, render_template, request
app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    In_data = {
        'tenure' : float(request.form['Tenure']),
        'age' : float(request.form['Age']),
        'address' : float(request.form['Address']),
        'income' : float(request.form['Income']),
        'ad' : float(request.form['Ad']),
        'employ' : float(request.form['Employ']),
        'longmon' : float(request.form['Longmon']),
        'tollmon' : float(request.form['Tollmon']),
        'eqipmon' : float(request.form['Equipmon']),
        'wiremon' : float(request.form['Wiremon']),
        'longten' : float(request.form['Longten']),
        'tollten' : float(request.form['Tollten']),
        'cardten' : float(request.form['Cardten']),
        'loglong' : float(request.form['Loglong']),
        'Ininc' : float(request.form['Ininc']),

    }

    In_df = pd.DataFrame([In_data])
    prediction = model.predict(In_df)[0]

    return render_template('result.html', predictions = prediction)



if(__name__)=='__main__':
    app.run(debug=True)





