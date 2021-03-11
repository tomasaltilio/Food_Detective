
from taxifare.transformers.utils import minkowski_distance

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd


class DistanceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, distance_type="euclidian", **kwargs):
        self.distance_type = distance_type

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.distance_type == "euclidian":
            X["distance"] = minkowski_distance(X, p=2)
        if self.distance_type == "manhattan":
            X["distance"] = minkowski_distance(X, p=1)
        return X[["distance"]]

    def fit(self, X, y=None):
        return self
