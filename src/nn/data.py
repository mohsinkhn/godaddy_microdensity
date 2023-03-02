import numpy as np
from torch.utils.data import Dataset


class GoData(Dataset):
    def __init__(self, data, seq_len=8, flag="train", trtype="test", start_month=18, labels=None):
        super().__init__()
        self.data = data.astype(np.float32).copy()
        self.seq_len = seq_len
        self.flag = flag
        self.trtype = trtype
        self.start_month = start_month
        self.labels = labels.astype(np.float32)
        self.data[np.isnan(self.data)] = 0

    def __len__(self):
        return self.data.shape[1]
    
    def __getitem__(self, idx):
        stidx, enidx = self.get_st_end()
        seq_data = self.data[stidx:enidx, idx, 0].astype(np.float32)
        norm_val = max(1, seq_data[-1])
        seq_data /= norm_val - 1
        if self.flag == "train":
            if self.trtype == "test":
                target = self.data[enidx+1:enidx+4, idx, 0]
                pop = np.log1p(self.data[enidx+1:enidx+4, idx, 1])
            else:
                target = self.data[enidx:enidx+1, idx, 0]
                pop = np.log1p(self.data[enidx:enidx+1, idx, 1])
            target /= norm_val - 1
        else:
            target = self.labels[:, idx, 0]
            target /= norm_val - 1
            pop = np.log1p(self.labels[:, idx, 1])

        return seq_data.reshape(-1, 1), target.astype(np.float32), pop, norm_val


def get_st_end(self):
    n = self.data.shape[0]
    if self.flag == "train":
        if self.trtype == "test":
            eval_len = 4
        else:
            eval_len = 1
        stidx = int(np.random.rand() * (n - eval_len - self.seq_len - self.start_month) + self.start_month)
        enidx = stidx + self.seq_len
    else:
        n = self.data.shape[0]
        stidx = n - self.seq_len
        enidx = n
    return stidx, enidx