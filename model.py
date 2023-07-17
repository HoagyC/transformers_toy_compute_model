import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

class Transformer(nn.Module):
    def __init__(self, d_resid, d_mlp_ratio = 4):
        super(Transformer, self).__init__()
        self.d_resid = d_resid
        self.d_mlp = d_resid * d_mlp_ratio
        self.l1 = nn.Sequential(nn.Linear(d_resid, self.d_mlp), nn.ReLU())
        self.l2 = nn.Linear(self.d_mlp, d_resid)

        torch.nn.init.xavier_uniform_(self.l1[0].weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        self.l1[0].bias.data.fill_(0.0)
        self.l2.bias.data.fill_(0.0)
    
    def forward(self, x):
        x_ = self.l1(x)
        x_ = self.l2(x_)
        x = x + x_
        return x


def train_model(in_features, target_features, dataset, steps = 5000, device="cuda"):
    model = Transformer(in_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i, batch in enumerate(itertools.islice(dataset, steps)):
        optimizer.zero_grad()
        resid_in = torch.einsum("ij,bi->bj", in_features, batch)
        resid_target = resid_in + torch.einsum("ij,bi->bj", target_features, batch)
        loss = F.mse_loss(model(resid_in), resid_target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(batch, resid_in, resid_target, loss)
        if i % 100 == 0:
            print(f"Step {i}: {loss.item()}")
    
    return model