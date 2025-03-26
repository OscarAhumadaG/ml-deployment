from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger


app = Flask(__name__)

app.config['SWAGGER'] = {
    'title': 'API Documentation',
    'uiversion': 3
}

swagger = Swagger(app)

pickle_in = open("logreg.pkl", "rb")
model = pickle.load(pickle_in)

@app.route('/')
def home():
    return "Welcome to the ML Prediction API!"


@app.route('/predict', methods=["GET"])
def predict_class():
    """Predict if Customer would buy the product or not.
    ---
    parameters:
      - name: age
        in: query
        type: number
        required: true
      - name: new_user
        in: query
        type: number
        required: true
      - name: total_pages_visited
        in: query
        type: number
        required: true
    responses:
      200:
        description: Prediction successful
      400:
        description: Invalid input
    """
    age = request.args.get("age")
    new_user = request.args.get("new_user")
    total_pages_visited = request.args.get("total_pages_visited")

    if age is None or new_user is None or total_pages_visited is None:
        return "❌ Missing parameters. Please provide age, new_user, and total_pages_visited.", 400

    try:
        age = int(age)
        new_user = int(new_user)
        total_pages_visited = int(total_pages_visited)
    except ValueError:
        return "❌ Invalid input. Parameters must be numbers.", 400

    prediction = model.predict([[age, new_user, total_pages_visited]])[0]

    return f"✅ Model prediction is {prediction}"


@app.route('/predict_file', methods=["POST"])
def prediction_test_file():
    """Predict from CSV file.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Prediction successful
      400:
        description: Invalid file input
    """
    try:
        df_test = pd.read_csv(request.files.get("file"))
        prediction = model.predict(df_test)
        return str([int(p) for p in prediction])
    except Exception as e:
        return f"❌ Error processing file: {str(e)}", 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


# http://127.0.0.1:5000/predict?age=30&new_user=1&total_pages_visited=5



