import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper

from src.constants import encoding_fields, scale_fields, selected_fields


class Preprocessor:
    mapper: DataFrameMapper

    def __init__(self):
        self.mapper = DataFrameMapper([
            (encoding_fields, [SimpleImputer(strategy="most_frequent"), preprocessing.OrdinalEncoder()]),
            (scale_fields, preprocessing.StandardScaler())])

    def train(self, x: pd.DataFrame):
        self.mapper.fit(x)

    def transform(self, x: pd.DataFrame):
        return self.mapper.transform(x)
