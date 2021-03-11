
import pandas as pd

from fastapi import FastAPI

import joblib

app = FastAPI()


@app.get("/")
def root():
    return {"greeting": "hello"}


@app.get("/predict")
def predict(pickup_datetime, lon1, lat1, lon2, lat2, passcount):

    # step 1 : convert params to dataframe
    X_pred = pd.DataFrame({
        "Unnamed: 0": ["0"],
        "key": ["truc"],
        "pickup_datetime": [pickup_datetime],
        "pickup_longitude": [float(lon1)],
        "pickup_latitude": [float(lat1)],
        "dropoff_longitude": [float(lon2)],
        "dropoff_latitude": [float(lat2)],
        "passenger_count": [int(passcount)]})

    # print(X_pred)
    # print(X_pred.columns)
    # print(X_pred.dtypes)

    # step 2 : load the trained model
    pipeline = joblib.load("model.joblib")
    # print(pipeline)

    # step 3 : make a prediction
    y_pred = pipeline.predict(X_pred)
    # print(type(y_pred))

    # step 4 : return the prediction (extract the prediction value from the ndarray)
    # print(y_pred)
    prediction = y_pred[0]

    return {"pred": prediction}
