from abc import abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils  import DataMatrix


class DMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dm: DataMatrix, fname: str):
        self.dm = dm
        self.fname = fname

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def agg(self, arr):
        return NotImplementedError("Implement an aggregation over numpy 2d array.")

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        df[self.dm.date_col] = pd.to_datetime(df[self.dm.date_col]).values
        df[self.fname] = np.nan
        dates = df[self.dm.date_col].unique()
        for date in dates:
            harr = self.dm.get_agg_arr(date)
            vals = self.agg(harr)
            indx = df[self.dm.date_col] == date
            items = df.loc[indx, self.dm.item_col].values
            item_vals = vals[np.array([self.dm.item2id[it] for it in items])]
            df.loc[indx, self.fname] = item_vals
        return df


class DMLastVal(DMTransformer):
    def agg(self, arr):
        return arr[-1, :, 0].flatten()


class DMLast2Val(DMTransformer):
    def agg(self, arr):
        y1 = arr[-1, :, 0]
        y2 = arr[-2, :, 0]
        y3 = arr[-3, :, 0]
        y4 = arr[-4, :, 0]
        y5 = arr[-5, :, 0]
        y6 = arr[-6, :, 0]
        w = np.clip(np.log1p(arr[-1, :, 0]*0.4), 0.0, 10)/10
        y = y1 + w * ((y1 - y2) * 0.25 + (y2 - y3) * 0.25 + (y3 - y4) * 0.25 + (y4 - y5) * 0.2 + (y5 - y6) * 0.05) + w * 0.5
        return y.flatten()
