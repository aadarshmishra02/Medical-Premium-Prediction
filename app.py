from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
model = pickle.load(open("Mlpro.pkl", "rb"))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = request.form['sex']
        smoker = request.form['smoker']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        region = request.form['region']

        # Convert categorical variables to numerical values
        sex_male = 1 if sex == 'male' else 0
        smoker_yes = 1 if smoker == 'yes' else 0
        region_northeast = 1 if region == 'northeast' else 0
        region_northwest = 1 if region == 'northwest' else 0
        region_southeast = 1 if region == 'southeast' else 0
        region_southwest = 1 if region == 'southwest' else 0

        # Create a numpy array with the input values
        values = np.array([[age, sex_male, smoker_yes, bmi, children, region_northeast, region_northwest, region_southeast, region_southwest]])

        # Make prediction using the model
        prediction = model.predict(values)
        prediction = round(prediction[0], 2)

        return render_template('result.html', prediction_text='Estimated medical insurance cost is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
