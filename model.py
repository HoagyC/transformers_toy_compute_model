import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools


class MLP(nn.Module):
    def __init__(self, d_resid, d_mlp):
        super(MLP, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(d_resid, d_mlp), nn.ReLU())
        self.l2 = nn.Linear(d_mlp, d_resid)

        torch.nn.init.xavier_uniform_(self.l1[0].weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        self.l1[0].bias.data.fill_(0.0)
        self.l2.bias.data.fill_(0.0)

    def forward(self, x):
        x_ = self.l1(x)
        x_ = self.l2(x_)
        return x_


class MLPOnlyTransformer(nn.Module):
    def __init__(self, d_resid, d_mlp_ratio: float = 4.0, n_mlps=1):
        super(MLPOnlyTransformer, self).__init__()
        self.d_resid = d_resid
        self.d_mlp = int(d_resid * d_mlp_ratio)
        self.mlps = nn.ModuleList([MLP(d_resid, self.d_mlp) for _ in range(n_mlps)])

    def forward(self, x):
        for mlp in self.mlps:
            x = x + mlp(x)
        return x


def train_model(in_features, target_features, mlp_ratio, dataset, steps=5000, device="cuda"):
    model = MLPOnlyTransformer(in_features.shape[1], d_mlp_ratio=mlp_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rolling_loss = 0.0
    for i, batch in enumerate(itertools.islice(dataset, steps)):
        optimizer.zero_grad()
        resid_in = torch.einsum("ij,bi->bj", in_features, batch)
        resid_target = resid_in + torch.einsum("ij,bi->bj", target_features, batch)
        loss = F.mse_loss(model(resid_in), resid_target)
        loss.backward()
        optimizer.step()
        if i == 0:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01
        if i % 1000 == 0:
            print(f"Step {i}: {loss.item()}")

    return model, rolling_loss
