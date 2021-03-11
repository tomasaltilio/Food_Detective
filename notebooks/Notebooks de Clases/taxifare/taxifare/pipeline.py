
from taxifare.transformers.distance_transformer import DistanceTransformer

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def get_pipeline(model):

    pipe_distance = make_pipeline(
        DistanceTransformer(),
        StandardScaler())

    cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

    feateng_blocks = [
        ('distance', pipe_distance, cols),
    ]

    features_encoder = ColumnTransformer(feateng_blocks)

    pipeline = Pipeline(steps=[
                ('features', features_encoder),
                ('model', model)])

    return pipeline
