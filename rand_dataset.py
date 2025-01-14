import torch
from typing import Any, Tuple, Optional, Union, Generator
from torchtyping import TensorType
import numpy as np

import os

from dataclasses import dataclass, field


@dataclass
class RandomDatasetGenerator(Generator):
    n_ground_truth_components: int
    batch_size: int
    feature_num_nonzero: int
    feature_prob_decay: float
    correlated: bool
    device: Union[torch.device, str]
    binary_feats: bool

    frac_nonzero: float = field(init=False)
    decay: TensorType["n_ground_truth_components"] = field(init=False)
    corr_matrix: Optional[TensorType["n_ground_truth_components", "n_ground_truth_components"]] = field(init=False)
    component_probs: Optional[TensorType["n_ground_truth_components"]] = field(init=False)

    def __post_init__(self):
        self.frac_nonzero = self.feature_num_nonzero / self.n_ground_truth_components

        # Define the probabilities of each component being included in the data
        self.decay = torch.tensor([self.feature_prob_decay**i for i in range(self.n_ground_truth_components)]).to(
            self.device
        )  # FIXME: 1 / i

        if self.correlated:
            self.corr_matrix = generate_corr_matrix(self.n_ground_truth_components, device=self.device)
        else:
            self.component_probs = self.decay * self.frac_nonzero  # Only if non-correlated
        self.t_type = torch.float32

    def send(self, ignored_arg: Any) -> TensorType["dataset_size", "activation_dim"]:
        if self.correlated:
            _, data = generate_correlated_dataset(
                self.n_ground_truth_components,
                self.batch_size,
                self.corr_matrix,
                self.frac_nonzero,
                self.decay,
                self.device,
                self.binary_feats,
            )
        else:
            _, data = generate_rand_dataset(
                self.n_ground_truth_components,
                self.batch_size,
                self.component_probs,
                self.device,
                self.binary_feats,
            )
        return data.to(self.t_type)

    def throw(self, type: Any = None, value: Any = None, traceback: Any = None) -> None:
        raise StopIteration


def generate_rand_dataset(
    n_ground_truth_components: int,  #
    dataset_size: int,
    feature_probs: TensorType["n_ground_truth_components"],
    device: Union[torch.device, str],
    binary_feats: bool = False,
):
    dataset_thresh = torch.rand(dataset_size, n_ground_truth_components, device=device)

    data_ones = torch.ones_like(dataset_thresh, device=device)
    data_zero = torch.zeros_like(dataset_thresh, device=device)

    dataset_codes = torch.where(
        dataset_thresh <= feature_probs,
        data_ones,
        data_zero,
    )  # dim: dataset_size x n_ground_truth_components

    if not binary_feats:
        # Multiply by a 2D random matrix of feature strengths
        feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
        dataset = dataset_codes * feature_strengths
    else:
        dataset = dataset_codes

    return dataset_codes, dataset


def generate_correlated_dataset(
    n_ground_truth_components: int,
    dataset_size: int,
    corr_matrix: TensorType["n_ground_truth_components", "n_ground_truth_components"],
    frac_nonzero: float,
    decay: TensorType["n_ground_truth_components"],
    device: Union[torch.device, str],
    binary_feats: bool,
):
    # Get a correlated gaussian sample
    mvn = torch.distributions.MultivariateNormal(
        loc=torch.zeros(n_ground_truth_components, device=device),
        covariance_matrix=corr_matrix,
    )
    corr_thresh = mvn.sample()

    # Take the CDF of that sample.
    normal = torch.distributions.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    cdf = normal.cdf(corr_thresh.squeeze())

    # Decay it
    component_probs = cdf * decay

    # Scale it to get the right % of nonzeros
    mean_prob = torch.mean(component_probs)
    scaler = frac_nonzero / mean_prob
    component_probs *= scaler
    # So np.isclose(np.mean(component_probs), frac_nonzero) will be True

    # Generate sparse correlated codes
    dataset_thresh = torch.rand(dataset_size, n_ground_truth_components, device=device)
    dataset_values = torch.rand(dataset_size, n_ground_truth_components, device=device)

    data_zero = torch.zeros_like(corr_thresh, device=device)
    dataset_codes = torch.where(
        dataset_thresh <= component_probs,
        dataset_values,
        data_zero,
    )
    # Ensure there are no datapoints w/ 0 features
    zero_sample_index = (dataset_codes.count_nonzero(dim=1) == 0).nonzero()[:, 0]
    random_index = torch.randint(low=0, high=n_ground_truth_components, size=(zero_sample_index.shape[0],)).to(
        dataset_codes.device
    )
    dataset_codes[zero_sample_index, random_index] = 1.0

    if not binary_feats:
        # Multiply by a 2D random matrix of feature strengths
        feature_strengths = torch.rand((dataset_size, n_ground_truth_components), device=device)
        dataset = dataset_codes * feature_strengths
    else:
        dataset = dataset_codes

    return dataset_codes, dataset


def generate_rand_feats(
    feat_dim: int,
    num_feats: int,
    device: Union[torch.device, str],
) -> TensorType["n_ground_truth_components", "activation_dim"]:
    feats = np.random.multivariate_normal(np.zeros(feat_dim), np.eye(feat_dim), size=num_feats)
    feats = feats.T / np.linalg.norm(feats, axis=1)
    feats = feats.T

    feats_tensor = torch.from_numpy(feats).to(device).float()
    return feats_tensor


def generate_corr_matrix(
    num_feats: int, device: Union[torch.device, str]
) -> TensorType["n_ground_truth_components", "n_ground_truth_components"]:
    # Create a correlation matrix
    corr_matrix = np.random.rand(num_feats, num_feats)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
    if min_eig < 0:
        corr_matrix -= 1.001 * min_eig * np.eye(corr_matrix.shape[0], corr_matrix.shape[1])

    corr_matrix_tensor = torch.from_numpy(corr_matrix).to(device).float()

    return corr_matrix_tensor
