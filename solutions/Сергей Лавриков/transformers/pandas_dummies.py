import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PandasDummies(BaseEstimator, TransformerMixin):
    def __init__(self, **params):
        self.params = params

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return pd.get_dummies(df, columns=list(set(self.params['cats']).intersection(df.columns)))

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params = params
        return self
