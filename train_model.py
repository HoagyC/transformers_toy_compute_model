from model import Transformer, train_model
from rand_dataset import RandomDatasetGenerator, generate_rand_feats

import torch

import itertools

import pickle

D_RESID = 512
N_FEATURES = 1024
D_MLP_RATIO = 4

BINARY_FEATS = False

if __name__ == "__main__":
    generator = RandomDatasetGenerator(
        N_FEATURES,
        1024,
        5,
        0.99,
        True,
        "cuda",
        BINARY_FEATS
    )

    initial_features = generate_rand_feats(D_RESID, N_FEATURES, "cuda")
    target_features = generate_rand_feats(D_RESID, N_FEATURES, "cuda")

    transformer_model = train_model(
        initial_features,
        target_features,
        generator,
        steps=5000
    )

    torch.save(transformer_model.state_dict(), "transformer.pt")
    with open("dataset.pt", "wb") as f:
        pickle.dump({"generator": generator, "input_feats": initial_features, "target_feats": target_features}, f)