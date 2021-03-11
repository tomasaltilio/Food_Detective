import pandas as pd

from TaxiFareModel.utils import simple_time_tracker
from google.cloud import storage

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


@simple_time_tracker
def get_data_from_gcp(nrows=10000, local=False, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    if local:
        path = "data/data_data_10Mill.csv"
    else:
        path = "gs://complete/correct_path/here"
    df = pd.read_csv(path, nrows=nrows)
    return df


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    df = get_data()
