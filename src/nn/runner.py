import argparse
from copy import deepcopy
from pathlib import Path


import numpy as np
from omegaconf import Omegaconf
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import wandb

from config import Fields, root_path, out_path
from src.last_value_baseline import anamoly_correction, get_splits
from src.nn.data import GoData
from src.nn.models import SeqModel
from src.nn.plmodel import LitModel
from src.utils import read_train_test_revealed, read_population, DataMatrix


seed = 457860
pl.seed_everything(seed)


def get_fold_data(df, fold_id):
    df_tr, df_vl = get_splits(df, fold_id, "test")
    dm = DataMatrix(F.date, F.cfips, key_cols=[F.active, F.md, F.pop])
    dm.fit(df_tr)

    for col in range(dm.arr.shape[1]):
        dm.arr[:, col, 0] = anamoly_correction(dm.arr[:, col, 0]) 
        
    vl_dates = df_vl[F.date].unique()
    arr_vl = np.zeros(shape=(len(vl_dates), dm.arr.shape[1], dm.arr.shape[2]), dtype=np.float32)
    for i, date in enumerate(vl_dates):
        tmp_df = df_vl.loc[df_vl[F.date] == date]
        cfips = tmp_df[F.cfips].values
        actives = tmp_df[F.active].values
        md = tmp_df[F.md].values
        pop = tmp_df[F.pop].values
        idx = [dm.item2id[cf] for cf in cfips]
        arr_vl[i, idx, 0] = actives
        arr_vl[i, idx, 1] = pop
        arr_vl[i, idx, 2] = md
    return dm.arr, arr_vl
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = args.config

    F = Fields()
    df = read_train_test_revealed()
    df["year"] = df[F.date].dt.year
    pop_df = read_population()
    df = pd.merge(df, pop_df, on=[F.cfips, "year"], how="left")
    assert df[F.pop].isnull().sum() == 0

    scores = []
    for fid in range(9):
        norm_arr_tr, norm_arr_vl = get_fold_data(df, fid)
        tr_ds = GoData(norm_arr_tr, seq_len=config.model.seq_len, flag="train", trtype="test", labels=None)
        tr_dl = DataLoader(tr_ds, batch_size=config.model.batch_size, shuffle=True, num_workers=config.model.num_workers)
        vl_ds = GoData(norm_arr_tr, seq_len=config.model.seq_len, flag="train", trtype="test", labels=norm_arr_vl)
        vl_dl = DataLoader(vl_ds, batch_size=2*config.model.batch_size, shuffle=False, num_workers=config.model.num_workers)
        
        model = LitModel(config, base_model=SeqModel(in_f=1, out_f=3, hidden_dim=config.model.hidden_dim, ))

        callbacks = [pl.callbacks.ModelCheckpoint(monitor='val/loss', save_top_k=2, mode='min', dirpath=f'data/checkpoints/{config.hidden_dim}_{seed}_{fid}_{version}', auto_insert_metric_name=False,
                                                filename="epoch={epoch}-val_loss={val/loss:.4f}")]
        logger = pl.loggers.wandb.WandbLogger(project='godaddy', name=f'nn_{fid}_{config.model.version}', tags=[f'{config.model.seq_len}', f'{config.model.hidden_dim}', f'fold_{fid}'], id=wandb.util.generate_id())
        logger.log_hyperparams(config)
        trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=config.model.max_epochs, logger=logger, callbacks=callbacks, deterministic=True)
        trainer.fit(model, tr_dl, vl_dl)
        score = trainer.callbacks[-1].best_model_score
        scores.append(score)
        wandb.finish()
    scores = np.array([sc.cpu().numpy() for sc in scores])
    print(np.mean(scores), np.std(scores), scores)



