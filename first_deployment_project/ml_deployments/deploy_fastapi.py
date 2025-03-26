import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class InputData(BaseModel):
    age: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int


with open("../models/adult_census_lr.pkl", "rb") as f:
    trained_model = pickle.load(f)


@app.post("/predict")
def predict(input_data: InputData):
    """
    Runs a prediction on the adult census data using a serialized logistic regression model.

    :param input_data: An instance of the InputData class containing values for age, capital-gain,
        capital-loss, and hours-per-week.
    :type input_data: InputData
    :return: A dictionary containing the predicted value.
    :rtype: dict
    """

    # Prepare input data as a pandas DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    input_df.columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

    # Make predictions
    predictions = trained_model.predict(input_df)

    # Return the predicted value as a dictionary
    return {"prediction": predictions[0]}


"""
Use curl correctly in PowerShell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Headers @{"accept"="application/json"; "Content-Type"="application/json"} -Body '{ "age": 43, "capital_gain": 14344, "capital_loss": 0, "hours_per_week": 40 }'

Execute in one line in PowerShell
curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d "{ \"age\": 43, \"capital_gain\": 14344, \"capital_loss\": 0, \"hours_per_week\": 40 }"

"""