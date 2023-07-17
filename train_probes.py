import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

from model import Transformer
from rand_dataset import RandomDatasetGenerator
from typing import Generator

import pickle

from dataclasses import dataclass, field

from typing import Dict

class MultiProbe(nn.Module):
    def __init__(self, n_features, d_activation):
        super(MultiProbe, self).__init__()
        self.n_features = n_features
        self.d_activation = d_activation
        self.probe = nn.Linear(d_activation, n_features, bias=False)
    
    def forward(self, x):
        # x: [batch_size, d_activation]
        return self.probe(x)
    

@dataclass
class ActivationDataset(Generator):
    transformer_model: Transformer
    name: str
    input_generator: RandomDatasetGenerator
    input_feats: torch.Tensor

    cache: Dict[str, torch.Tensor] = field(init=False)

    def __post_init__(self):
        def get_act(cache, name):
            def hook(model, input, output):
                cache[name] = output
            return hook

        self.cache = {}
        self.transformer_model.get_submodule(self.name).register_forward_hook(get_act(self.cache, self.name))
    
    def send(self, ignored_arg):
        x = self.input_generator.__next__()
        resid_in = torch.einsum("ij,bi->bj", self.input_feats, x)
        self.transformer_model(resid_in)
        return (x, self.cache[self.name])
    
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

def train_probes(transformer_model, probe, dataset, input_feats, steps=5000):
    act_dataset = ActivationDataset(transformer_model, "l1", dataset, input_feats)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for i, (dict_scores, acts) in enumerate(itertools.islice(act_dataset, steps)):
        optimizer.zero_grad()
        preds = probe(acts)
        loss = F.mse_loss(dict_scores, preds)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Step {i}: {loss.item()}")
    
    return probe

if __name__ == "__main__":
    with open("dataset.pt", "rb") as f:
        data = pickle.load(f)
        generator = data["generator"]
        input_features = data["input_feats"]
        target_features = data["target_feats"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer_state_dict = torch.load("transformer.pt")
    transformer_model = Transformer(input_features.shape[1]).to(device)
    transformer_model.load_state_dict(transformer_state_dict)

    MLP_WIDTH = transformer_model.l2.in_features
    N_FEATURES = input_features.shape[0]

    print(MLP_WIDTH, N_FEATURES)

    probe = MultiProbe(N_FEATURES, MLP_WIDTH).to(device)

    print(probe.probe)

    probe = train_probes(transformer_model, probe, generator, input_features, steps=5000)

    torch.save(probe.state_dict(), "probe.pt")