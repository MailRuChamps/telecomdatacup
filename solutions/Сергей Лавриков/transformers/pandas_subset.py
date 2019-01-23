from sklearn.base import BaseEstimator, TransformerMixin


class PandasSubset(BaseEstimator, TransformerMixin):
    def __init__(self, **params):
        self.params = params

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[list(set(self.fields()) & set(df.columns))]

    def fields(self):
        return [k for k, v in self.params.items() if v]

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params = params
        return self
