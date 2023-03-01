import numpy as np
import pandas as pd
from pathlib import Path

from config import root_path, Fields


def read_train_test_revealed():
    F = Fields()
    train = pd.read_csv(Path(root_path) / "train.csv")
    test_revealed = pd.read_csv(Path(root_path) / "revealed_test.csv")
    train = pd.concat([train, test_revealed], axis=0).reset_index(drop=True)
    train[F.date] = pd.to_datetime(train[F.date])
    return train.sort_values(by=[F.cfips, F.date])


def read_population():
    return pd.read_csv(Path(root_path) / "populations.csv")


class DataMatrix:
    def __init__(self, date_col, item_col, key_cols):
        self.date_col = date_col
        self.item_col = item_col
        self.key_cols = key_cols
        self.arr = None
        self.date2id, self.id2date = None, None
        self.item2id, self.id2item = None, None

    def fit(self, df: pd.DataFrame):
        F = Fields()
        df = df.sort_values(by=[self.item_col, self.date_col])
        self.dates = df[F.date].unique()
        self.items = df[F.cfips].unique()
        self.date2id = {v: i for i, v in enumerate(self.dates)}
        self.id2date = {v: k for k, v in self.date2id.items()}
        self.item2id = {v: i for i, v in enumerate(self.items)}
        self.id2item = {v: k for k, v in self.item2id.items()}

        p, q, r = len(self.dates), len(self.items), len(self.key_cols)
        self.arr = np.zeros(shape=(p, q, r), dtype=np.float32)
        dts = df[F.date].map(self.date2id).values
        cfips = df[F.cfips].map(self.item2id).values

        for i, col in enumerate(self.key_cols):
            data =  df[col].values
            self.arr[dts, cfips, i] = data
        return

    def get_agg_arr(self, date):
        if date in self.dates:
            return self.arr[:self.date2id[date]]
        else:
            idx = np.searchsorted(self.dates, date)
            return self.arr[:idx]
