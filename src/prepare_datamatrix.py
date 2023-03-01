import pandas as pd
import numpy as np

from config import Fields
from src.utils import read_train_test_revealed, read_population


if __name__ == "__main__":
    F = Fields()
    df = read_train_test_revealed()

    df["year"] = df[F.date].dt.year
    pop_df = read_population()
    df = pd.merge(df, pop_df, on=[F.cfips, "year"], how="left")
    assert df[F.pop].isnull().sum() == 0

    df.pivot(index=F.date, columns=F.cfips, values=[F.active, F.md, F.pop])
    
