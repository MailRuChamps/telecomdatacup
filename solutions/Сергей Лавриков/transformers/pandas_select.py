from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class PandasSelect(BaseEstimator, TransformerMixin):
    def __init__(self, field, dtype=None, fillna_zero=False):
        self.field = field
        self.dtype = dtype
        self.fillna_zero = fillna_zero

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        if self.field in dataframe.columns:
            dt = dataframe[self.field].dtype
            if is_categorical_dtype(dt):
                return dataframe[self.field].cat.codes[:, None]
            elif is_numeric_dtype(dt):
                if self.dtype is not None:
                    return dataframe[self.field].astype(self.dtype)[:, None]
                else:
                    return dataframe[self.field][:, None]
            else:
                return dataframe[self.field]
        elif self.fillna_zero:
            if self.dtype is not None:
                return np.zeros((dataframe.index.shape[0], 1), dtype=self.dtype)
            else:
                return np.zeros((dataframe.index.shape[0], 1), dtype=np.float16)
        else:
            return dataframe[self.field][:, None]
