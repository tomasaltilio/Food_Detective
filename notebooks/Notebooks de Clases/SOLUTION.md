
## Steps

### Create a trained model from the package of the solution

Run the model locally:

``` bash
python -m TaxiFareModel.trainer
```

That will create a `model.joblib` file, which we will use for our API.

### Run our API locally

``` bash
make run_api
```

Connect to the root endpoint: http://127.0.0.1:8000

You should see the response:

```
{
  "ok": "True"
}
```

Connect to the prediction endpoint: http://127.0.0.1:8000/predict_fare/?key=2012-10-06%2012:10:20.0000001&pickup_datetime=2012-10-06%2012:10:20%20UTC&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2

You should see a prediction:

```
{
  "prediction": 36.7248779841608
}
```

Let's do the same thing using our Notebook (in another terminal window, since the API server must continue to run):

``` bash
jupyter notebook
```

Open the `API usage` notebook and run the cell...

You should see a prediction:

```
{'prediction': 35.45436628928188}
```
