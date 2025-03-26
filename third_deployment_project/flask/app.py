import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_prediction = None  # Ensure there is always a default value

    if request.method == 'POST':
        try:
            # Capture the input values and convert them to float
            var_1 = float(request.form.get('var_1', 0))
            var_2 = float(request.form.get('var_2', 0))
            var_3 = float(request.form.get('var_3', 0))
            var_4 = float(request.form.get('var_4', 0))
            var_5 = float(request.form.get('var_5', 0))

            # Convert to a numpy array
            pred_args = [var_1, var_2, var_3, var_4, var_5]
            preds = np.array(pred_args).reshape(1, -1)

            # Load the model correctly using 'with open'
            with open("../model/linear_regression_model.pkl", "rb") as model_file:
                lr_model = joblib.load(model_file)

            # Make the prediction
            model_prediction = round(float(lr_model.predict(preds)[0]), 2)

        except ValueError:
            return "Please enter valid numeric values."

    return render_template('predict.html', prediction=model_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


