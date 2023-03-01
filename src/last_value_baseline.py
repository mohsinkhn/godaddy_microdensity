import argparse
import datetime

from logzero import logger
import numpy as np
import pandas as pd

from config import root_path, Fields, out_path
from src.eval import smape
from src.sktransformers import DMLastVal, DMLast2Val
from src.utils import DataMatrix, read_train_test_revealed, read_population


pd.options.mode.chained_assignment = None  # default='warn'


def anamoly_correction(ar):
    arr = ar[:]
    diffs = np.diff(arr)
    med = np.median(np.abs(diffs))
    p25 = np.percentile(diffs, 20)
    p75 = np.percentile(diffs, 80)
    iqr = p75 - p25
    upper_limit = med + 3 * iqr
    anam = np.where(np.abs(diffs) > upper_limit)[0]
    if len(anam) > 0:
        max_idx = np.max(anam)
        arr[:max_idx+1] = arr[max_idx+1]
        # arr[:max_idx+1] = arr[:max_idx+1] + diffs[max_idx] * 1.0
    return arr


def get_splits(df, fid, eval_method):
    F = Fields()
    dates = sorted(df[F.date].unique())
    n = len(dates)
    if eval_method == "public":
        split_date = dates[-fid-1]
        logger.debug(f"Fold: {fid} ; Split Date: {np.datetime_as_string(split_date, unit='D')}")
        df_tr, df_vl = df.loc[df[F.date] < split_date], df.loc[df[F.date] == split_date]
    else:
        p = n - fid - 4
        split_date = dates[p]
        vl_dates = dates[p+1:p+4]
        logger.debug(f"Fold: {fid} ; Split Date: {np.datetime_as_string(split_date, unit='D')}")
        df_tr, df_vl = df.loc[df[F.date] < split_date], df.loc[df[F.date].isin(vl_dates)]
    return df_tr, df_vl


def score_a_fold(df_tr, df_vl):
    dm = DataMatrix(F.date, F.cfips, key_cols=[F.active, F.md, F.pop])
    dm.fit(df_tr)
    for col in range(dm.arr.shape[1]):
        dm.arr[:, col, 0] = anamoly_correction(dm.arr[:, col, 0]) 
    tfm = DMLast2Val(dm, fname='last_val')
    df_vl = tfm.transform(df_vl)
    df_vl['preds'] = df_vl.loc[:, 'last_val'] / df_vl.loc[:, F.pop] * 100
    return tfm, df_tr, df_vl


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default=9)
    parser.add_argument("--eval_method", default="public")

    args = parser.parse_args()

    n_folds = args.folds
    eval_method = args.eval_method

    F = Fields()
    df = read_train_test_revealed()

    df["year"] = df[F.date].dt.year
    pop_df = read_population()
    df = pd.merge(df, pop_df, on=[F.cfips, "year"], how="left")
    assert df[F.pop].isnull().sum() == 0
    #  N Folds
    scores = []
    for fid in range(n_folds):
        df_tr, df_vl = get_splits(df, fid, eval_method)
        logger.debug(f"Train: {df_tr.shape}, Val: {df_vl.shape}")
        # fit and infer a fold
        model, df_tr, df_vl = score_a_fold(df_tr, df_vl)
        # score
        score = smape(df_vl[F.md].values, df_vl["preds"].values)
        logger.info(f"Score for fold {fid} - {score: 6.4f}")
        scores.append(score)

    logger.info(f"Mean score across all folds {np.mean(scores)} with std: {np.std(scores)}")
