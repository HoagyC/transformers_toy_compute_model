from model import Transformer, train_model
from rand_dataset import RandomDatasetGenerator, generate_rand_feats

import torch

import itertools

import pickle

D_RESID = 500
N_FEATURES = 1000
D_MLP_RATIO = 4

BINARY_FEATS = False

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    generator = RandomDatasetGenerator(
        n_ground_truth_components=N_FEATURES,
        batch_size=1024,
        feature_num_nonzero=5,
        feature_prob_decay=0.99,
        correlated=False,
        device=device,
        binary_feats=BINARY_FEATS
    )

    initial_features = generate_rand_feats(D_RESID, N_FEATURES, device)
    target_features = generate_rand_feats(D_RESID, N_FEATURES, device)

    transformer_model = train_model(
        initial_features,
        target_features,
        generator,
        steps=5000,
        device=device
    )

    torch.save(transformer_model.state_dict(), "transformer.pt")
    with open("dataset.pt", "wb") as f:
        pickle.dump({"generator": generator, "input_feats": initial_features, "target_feats": target_features}, f)