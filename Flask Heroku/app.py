# Importing the necessary libraries
from flask import Flask, request, render_template
from flask_cors import CORS,cross_origin
import numpy as np
import pickle

app = Flask(__name__)  # Initialising flask app


@app.route('/', methods=['GET'])  # route to display the Home page
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in web UI
@cross_origin()
def prediction():
    if request.method == 'POST':
        # reading the inputs given by the user
        age = int(request.form['age'])
        hours = int(request.form['hours'])
        gain = int(request.form['gain'])
        loss = int(request.form['loss'])
        sex = request.form['sex']
        marital = request.form['marital']
        country = request.form['country']
        education = request.form['education']

        sex = 1 if sex == 'Male' else 0
        marital = 0 if marital == 'Married' else 1
        country = 1 if country == 'United States' else 0

        # if condition match only assign '1' to that variable
        _11th = 1 if education == '_11th' else 0
        _12th = 1 if education == '_12th' else 0
        _1st_4th = 1 if education == '_1st_4th' else 0
        _5th_6th = 1 if education == '_5th_6th' else 0
        _7th_8th = 1 if education == '_7th_8th' else 0
        _9th = 1 if education == '_9th' else 0
        _Assoc_acdm = 1 if education == 'Assoc_acdm' else 0
        assoc_voc = 1 if education == 'assoc_voc' else 0
        bachelors = 1 if education == 'bachelors' else 0
        doctorate = 1 if education == 'doctorate' else 0
        HS_grad = 1 if education == 'HS_grad' else 0
        masters = 1 if education == 'masters' else 0
        preschool = 1 if education == 'preschool' else 0
        prof_school = 1 if education == 'prof_school' else 0
        college = 1 if education == 'college' else 0

        # load the model
        model = pickle.load(open('xgb_model.pkl', 'rb'))
        # load the scaler
        scaler = pickle.load(open('scaler.pkl', 'rb'))

        # feature scaling on age,capital_gain, capital_loss, hours per week
        scaled_value = scaler.transform([[age, gain, loss, hours]])
        age, gain, loss, hours = scaled_value[0, 0], scaled_value[0, 1], scaled_value[0, 2], scaled_value[0, 3]

        # predictions using the loaded model file
        predict = model.predict(np.array([[age, hours, gain, loss, _11th, _12th, _1st_4th, _5th_6th, _7th_8th, _9th,
                                           _Assoc_acdm, assoc_voc, bachelors, doctorate, HS_grad, masters, preschool,
                                           prof_school, college, marital, sex, country]]))[0]
        output = "Annual Income is More Than 50K" if predict == 1 else "Annual Income is Less Than 50K"

        # showing the prediction result in a UI
        return render_template('result.html', output=output)


if __name__ == '__main__':
    app.run(debug=True)
