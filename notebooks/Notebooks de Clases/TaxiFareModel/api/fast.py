
import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict_fare/?key=2012-10-06 12:10:20.0000001&pickup_datetime=2012-10-06 12:10:20 UTC&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2


@app.get("/")
def index():
    return {"ok": "True"}


@app.get("/predict_fare/")
def create_fare(key,
                pickup_datetime,
                pickup_longitude,
                pickup_latitude,
                dropoff_longitude,
                dropoff_latitude,
                passenger_count):

    # key = "2013-07-06 17:18:00.000000119"
    # pickup_datetime = "2013-07-06 17:18:00 UTC"
    # pickup_longitude = "-73.950655"
    # pickup_latitude = "40.783282"
    # dropoff_longitude = "-73.984365"
    # dropoff_latitude = "40.769802"
    # passenger_count = "1"

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
        key=[key],
        pickup_datetime=[pickup_datetime],
        pickup_longitude=[float(pickup_longitude)],
        pickup_latitude=[float(pickup_latitude)],
        dropoff_longitude=[float(dropoff_longitude)],
        dropoff_latitude=[float(dropoff_latitude)],
        passenger_count=[int(passenger_count)]))

    # ⚠️ TODO: get model from GCP

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(
        prediction=pred)
