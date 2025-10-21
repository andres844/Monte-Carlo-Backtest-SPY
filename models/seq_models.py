import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, p=0.1):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # x: [B, C, T]
        y = self.net(x)
        crop = y.shape[-1] - x.shape[-1]
        if crop > 0:
            y = y[..., :-crop]
        return y + self.res(x)


class TCNModel(nn.Module):
    def __init__(self, in_dim, channels=(32, 32, 32), dilations=(1, 2, 4), p=0.1):
        super().__init__()
        layers = []
        c_in = in_dim
        for c, d in zip(channels, dilations):
            layers.append(TCNBlock(c_in, c, k=3, d=d, p=p))
            c_in = c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x):  # x: [B, T, F]
        x = x.transpose(1, 2)
        y = self.tcn(x)
        last = y[:, :, -1]
        return self.head(last).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=1, p=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.0 if layers == 1 else p,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: [B, T, F]
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


class TCNBackbone(nn.Module):
    """
    TCN feature extractor that returns the last-step representation.
    """
    def __init__(self, in_dim, channels=(32, 32, 32), dilations=(1, 2, 4), p=0.1):
        super().__init__()
        layers = []
        c_in = in_dim
        for c, d in zip(channels, dilations):
            layers.append(TCNBlock(c_in, c, k=3, d=d, p=p))
            c_in = c
        self.tcn = nn.Sequential(*layers)
        self.out_dim = c_in

    def forward(self, x):  # x: [B, T, F]
        x = x.transpose(1, 2)
        y = self.tcn(x)
        last = y[:, :, -1]
        return last  # [B, C]


class LSTMBackbone(nn.Module):
    """
    LSTM feature extractor that returns the final hidden state.
    """
    def __init__(self, in_dim, hidden=64, layers=1, p=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=0.0 if layers == 1 else p)
        self.out_dim = hidden

    def forward(self, x):  # x: [B, T, F]
        _, (h, _) = self.lstm(x)
        return h[-1]  # [B, H]


class DualHeadModel(nn.Module):
    """
    Dual-head model for directional Kelly sizing.
    Heads:
      - p_head: probability of positive return (logit)
      - pos_head: expected positive magnitude E[r+]
      - neg_head: expected negative magnitude E[|r-|]
    """
    def __init__(self, in_dim, backbone: str = 'tcn', **kwargs):
        super().__init__()
        if backbone == 'tcn':
            self.backbone = TCNBackbone(in_dim, **kwargs)
        elif backbone == 'lstm':
            self.backbone = LSTMBackbone(in_dim, **kwargs)
        else:
            raise ValueError("backbone must be 'tcn' or 'lstm'")
        h = self.backbone.out_dim
        self.p_head = nn.Linear(h, 1)
        self.pos_head = nn.Linear(h, 1)
        self.neg_head = nn.Linear(h, 1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, T, F]
        h = self.backbone(x)
        logit = self.p_head(h)
        p = self.sigmoid(logit).squeeze(-1)
        pos = self.softplus(self.pos_head(h)).squeeze(-1)
        neg = self.softplus(self.neg_head(h)).squeeze(-1)
        return {'p': p, 'pos': pos, 'neg': neg, 'logit': logit.squeeze(-1)}
