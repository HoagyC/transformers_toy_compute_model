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

def computation_objective(in_features, target_features):
    # in_features: [n_features, d_resid]
    # target_features: [n_features, d_resid]
    def go(model, x):
        # x: [batch_size, n_features]
        resid_in = torch.einsum("ij,bi->bj", in_features, x)
        resid_target = resid_in + torch.einsum("ij,bi->bj", target_features, x)
        return F.mse_loss(model(resid_in), resid_target)
    
    return go

def train_model(in_features, target_features, dataset, steps = 5000):
    model = Transformer(in_features.shape[1]).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = computation_objective(in_features, target_features)
    for i, batch in enumerate(itertools.islice(dataset, steps)):
        optimizer.zero_grad()
        loss = lossfn(model, batch)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Step {i}: {loss.item()}")
    
    return model