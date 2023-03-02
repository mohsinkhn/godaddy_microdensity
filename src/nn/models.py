import torch
from torch import nn


class SeqModel(nn.Module):
    def __init__(
        self,
        in_f,
        out_f,
        seq_l,
        hidden_dim,
        dropout=0.1,
        num_layers=1,
        rnn_bias=True,
        bidirectional=False,
        linear_bias=True,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            in_f,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bias=rnn_bias,
            bidirectional=bidirectional,
        )
        self.out = nn.Linear(hidden_dim * seq_l, out_f, bias=linear_bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pop):
        b, _, _ = x.shape
        x, _ = self.rnn(x)
        x = x.reshape(b, -1).contiguous()
        x = self.out(self.drop(x))
        return x


class SeqModel2(nn.Module):
    def __init__(
        self,
        in_f,
        out_f,
        seq_l,
        hidden_dim,
        dropout=0.1,
        num_layers=1,
        rnn_bias=True,
        bidirectional=False,
        linear_bias=True,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            in_f + 1,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bias=rnn_bias,
            bidirectional=bidirectional,
        )
        self.out = nn.Linear(hidden_dim * seq_l, out_f, bias=linear_bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pop):
        b, l, _ = x.shape
        x2 = torch.log1p(pop.unsqueeze(1).repeat(1, l, 1)) / 100
        x = torch.cat([x, x2], axis=-1)
        x, _ = self.h1(x)
        x = x.reshape(b, -1).contiguous()
        x = self.out(self.drop2(x))
        return x
