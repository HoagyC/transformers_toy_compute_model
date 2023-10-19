from typing import List, Tuple

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@st.cache_data
def compute_feature_vectors(vector_dim, dims_per_feature, n_feats):
    """Generate a set of feature vectors, each of which is a unit vector with dims_per_feature non-zero dimensions"""
    feature_vectors = torch.rand(n_feats, vector_dim) - 0.5
    for i in range(n_feats):
        dead_feats = np.random.choice(vector_dim, vector_dim - dims_per_feature, replace=False)
        feature_vectors[i, dead_feats] = 0
        feature_vectors[i] /= feature_vectors[i].norm()
    return feature_vectors


def generate_dataset(n_samples, n_feats, sparsity, feature_vectors, nonlin_fn, scale, bias, always_on_ndxs=None):
    """Generate a sparse dataset with n_samples samples and n_feats features, where each feature is a linear combination of the feature vectors"""
    if type(sparsity) == int:
        is_on_matrix = torch.zeros(n_samples, n_feats).to(torch.bool)
        for s in range(n_samples):
            on_ndxs = torch.randperm(n_feats)[:sparsity]
            is_on_matrix[s, on_ndxs] = True
    else:
        is_on_matrix = torch.rand(n_samples, n_feats) > sparsity

    if always_on_ndxs:
        is_on_matrix[:, always_on_ndxs] = 1
    negative_vector_levels = torch.zeros(n_samples, n_feats)
    positive_vector_levels = torch.rand(n_samples, n_feats) * scale
    vector_levels = torch.where(is_on_matrix, positive_vector_levels, negative_vector_levels)
    inputs = vector_levels @ feature_vectors
    outputs = nonlin_fn(inputs + bias)
    return inputs, outputs, vector_levels


def plot_response_curves(probed_input, probed_output, vector_levels, feature_ndx):
    """Plot the response curves for a single feature"""
    fig, ax = plt.subplots(1, 1)
    ax.scatter(vector_levels[:, feature_ndx], probed_input[:, feature_ndx], label="input", color="red")
    ax.scatter(vector_levels[:, feature_ndx], probed_output[:, feature_ndx], label="output")
    ax.set_title("Response curve for a single feature")
    ax.legend()
    ax.set_xlabel("input level")
    ax.set_ylabel("output level")
    return fig


def binary_relation_metric(vector1, vector2, test_prop):
    """Testing how well the relationship between two vectors can be modelled as a binary variable"""
    # Reshape the vectors to be 2D arrays
    vector1 = vector1.reshape(-1, 1)
    vector2 = vector2.reshape(-1, 1)

    # Binarize vector2
    median_val = np.median(vector2)
    vector2_binary = (vector2 > median_val).astype(int)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(vector1, vector2_binary, test_size=test_prop, random_state=0)

    # Fit logistic regression model
    clf = LogisticRegression()
    clf.fit(X_train, y_train.ravel())

    # Predict probabilities
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Compute AUC-ROC
    auc_roc = roc_auc_score(y_test, y_prob)
    return auc_roc


def measure_binary_nonlinearity(vector_levels, probed_input, probed_output, feature_ndx, test_prop=0.1):
    """
    Want to check whether there's a particular threshold between which the input-output relationship
    can be modelled as a binary variable.
    """
    feat_vector_levels = vector_levels[:, feature_ndx].numpy()
    feat_probed_output = probed_output[:, feature_ndx].numpy()
    feat_probed_input = probed_input[:, feature_ndx].numpy()

    input_binary_score = binary_relation_metric(feat_vector_levels, feat_probed_input, test_prop)
    output_binary_score = binary_relation_metric(feat_vector_levels, feat_probed_output, test_prop)

    return output_binary_score - input_binary_score


def measure_nonlinearity(vector_levels, probed_output, feature_ndx, test_prop=0.9):
    assert vector_levels.shape == probed_output.shape
    n_samples = probed_output.shape[0]
    test_begin_ndx = int(n_samples * (1 - test_prop))
    quad_reg_inputs = torch.stack([vector_levels, vector_levels**2], dim=1)

    lin_reg = LinearRegression()
    lin_reg.fit(vector_levels[:test_begin_ndx, feature_ndx].unsqueeze_(1), probed_output[:test_begin_ndx, feature_ndx])
    lin_reg_score = lin_reg.score(
        vector_levels[test_begin_ndx:, feature_ndx].unsqueeze_(1), probed_output[test_begin_ndx:, feature_ndx]
    )

    quad_reg = LinearRegression()
    quad_reg = quad_reg.fit(
        quad_reg_inputs[:test_begin_ndx, :, feature_ndx], probed_output[:test_begin_ndx, feature_ndx]
    )
    quad_reg_score = quad_reg.score(
        quad_reg_inputs[test_begin_ndx:, :, feature_ndx], probed_output[test_begin_ndx:, feature_ndx]
    )
    return quad_reg_score - lin_reg_score


def main():
    st.title("Streamlit for Neural Network Exploration")

    # Sidebar
    vector_dim = st.sidebar.slider("Vector Dimension", 5, 200, 100)
    dims_per_feature = st.sidebar.slider("Dims per Feature", 1, vector_dim, 3)
    n_feats = st.sidebar.slider("Number of Features", 50, 2000, 100)
    sparsity = st.sidebar.slider("Sparsity", 1, n_feats, 5, 1)
    nonlin = st.sidebar.selectbox("Nonlinearity", ["relu", "gelu", "tanh"])
    bias = st.sidebar.slider("Bias", -10.0, 10.0, 0.0, 0.1)
    scale = st.sidebar.slider("Scale", 0.5, 50.0, 5.0, 0.5)
    feature_ndx = st.sidebar.slider("Feature Index for Plotting", 0, n_feats - 1, 0)

    nonlin_dict = {"relu": torch.nn.ReLU(), "gelu": torch.nn.GELU(), "tanh": torch.nn.Tanh()}
    nonlin_options = ["relu", "gelu", "tanh"]
    nonlin = st.sidebar.selectbox("Select Nonlinearity:", nonlin_options)
    nonlin_fn = nonlin_dict[nonlin]

    feature_vectors = compute_feature_vectors(vector_dim, dims_per_feature, n_feats)

    # Main Content
    st.write("## Dataset Generation")
    n_samples = 5000
    inputs, outputs, vector_levels = generate_dataset(
        n_samples, n_feats, sparsity, feature_vectors, nonlin_fn, scale, bias, always_on_ndxs=[feature_ndx]
    )

    # Plotting
    st.write("## Response Curves")
    probed_input = inputs @ feature_vectors.T
    probed_output = outputs @ feature_vectors.T
    fig = plot_response_curves(probed_input, probed_output, vector_levels, feature_ndx)
    st.pyplot(fig)

    nonlins = np.zeros((8, 10))
    print(nonlins.shape)
    print(nonlins[2, 2])
    for i, scale in enumerate([2**x for x in range(10)]):
        for j, bias in enumerate([-(2**x) for x in range(8)]):
            n_samples = 500
            inputs, outputs, vector_levels = generate_dataset(
                n_samples, n_feats, sparsity, feature_vectors, nonlin_fn, scale, bias, always_on_ndxs=[feature_ndx]
            )
            probed_output = outputs @ feature_vectors.T
            nonlins[j, i] = measure_nonlinearity(vector_levels, probed_output, feature_ndx)

    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.imshow(nonlins, vmin=0, vmax=0.2)
    ax.set_xlabel("Feature input scale")
    ax.set_ylabel("MLP Bias")
    ax.set_title("Nonlinearity measure by feature scale and mlp bias.")
    ax.set_xticklabels([2**x for x in range(10)])
    ax.set_yticklabels([-1] + [-(2**x) for x in range(8)])
    fig.colorbar(cax, shrink=0.3, aspect=15 * 0.5)
    st.pyplot(plt, use_container_width=True)

    st.write(f"Max non-linearity score: {np.max(nonlins):.2}")


if __name__ == "__main__":
    main()
