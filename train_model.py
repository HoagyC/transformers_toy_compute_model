from model import MLPOnlyTransformer, train_model
from rand_dataset import RandomDatasetGenerator, generate_rand_feats
from train_probes import MultiProbe, train_probes

import matplotlib.pyplot as plt
import torch

from datetime import datetime
import itertools

import pickle

BINARY_FEATS = False


def train_one():
    d_resid = 500
    n_feats = 1000
    mlp_ratio = 1

    generator = RandomDatasetGenerator(
        n_ground_truth_components=n_feats,
        batch_size=1024,
        feature_num_nonzero=5,
        feature_prob_decay=0.99,
        correlated=False,
        device=device,
        binary_feats=BINARY_FEATS,
    )

    initial_features = generate_rand_feats(d_resid, n_feats, device)
    target_features = generate_rand_feats(d_resid, n_feats, device)

    transformer_model, rolling_loss = train_model(
        initial_features, target_features, generator, steps=1000, device=device
    )

    torch.save(transformer_model.state_dict(), "transformer.pt")
    with open("dataset.pt", "wb") as f:
        pickle.dump(
            {
                "generator": generator,
                "input_feats": initial_features,
                "target_feats": target_features,
            },
            f,
        )


def train_multiple(device):
    mlp_ratio = 1
    d_resid_range = [2**x for x in range(4, 9)]
    feat_ratio_range = [1.5**x for x in range(0, 5)]

    d_resid_losses = []
    print(f"Training on device={device} with d_resid_range={d_resid_range} and feat_ratio_range={feat_ratio_range}")
    for d_resid in d_resid_range:
        losses = []
        for feat_ratio in feat_ratio_range:
            n_feats = int(d_resid * feat_ratio)
            print(f"Training for d_resid={d_resid}, feat_ratio={feat_ratio}, n_feats={n_feats}")
            generator = RandomDatasetGenerator(
                n_ground_truth_components=n_feats,
                batch_size=1024,
                feature_num_nonzero=5,
                feature_prob_decay=1,
                correlated=True,
                device=device,
                binary_feats=BINARY_FEATS,
            )

            initial_features = generate_rand_feats(d_resid, n_feats, device)
            target_features = generate_rand_feats(d_resid, n_feats, device)

            transformer_model, rolling_loss = train_model(
                initial_features,
                target_features,
                mlp_ratio,
                generator,
                steps=15000,
                device=device,
            )
            print(f"Final loss: {rolling_loss}")

            # training probes to see how well the model is able to linearly separate the features
            probe = MultiProbe(n_feats, transformer_model.d_mlp).to(device)

            probe = train_probes(transformer_model, probe, generator, initial_features, steps=10000)

            losses.append(rolling_loss)
        d_resid_losses.append(losses)

    # save the data
    with open("losses.pt", "wb") as f:
        pickle.dump(
            {
                "d_resid_range": d_resid_range,
                "feat_ratio_range": feat_ratio_range,
                "d_resid_losses": d_resid_losses,
            },
            f,
        )

    # plot the two loss lines on the same graph
    for i, d_resid in enumerate(d_resid_range):
        plt.plot(feat_ratio_range, d_resid_losses[i], label=f"d_resid={d_resid}")
    plt.legend()
    # log scale on both axes
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("feat_ratio")
    plt.ylabel("loss")
    # save
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"losses_{time}.png")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_multiple(device)
