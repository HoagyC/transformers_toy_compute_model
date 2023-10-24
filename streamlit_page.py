import logging
from typing import List, Tuple

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@st.cache_data
def compute_feature_vectors(vector_dim, dims_per_feature, n_feats):
    """Generate a set of feature vectors, each of which is a unit vector with dims_per_feature non-zero dimensions"""
    feature_vectors = torch.rand(n_feats, vector_dim) - 0.5
    for i in range(n_feats):
        dead_feats = np.random.choice(vector_dim, vector_dim - dims_per_feature, replace=False)
        feature_vectors[i, dead_feats] = 0
        feature_vectors[i] /= feature_vectors[i].norm()
    return feature_vectors


def test_linearity(feature_vectors, feature_ndxs: List[int], n_samples, feat_min=0, feat_max=5, test_frac=0.2):
    """
    So conceptually, we're taking the features that are coming in, and testing whether we find that there's linearity there.
    One possible form, we take a set of features that co-occur, or co-occur quite regularly, and ask whether we see linearity amongst this basis
    whre this takes the form of asking whether we see if f(ax + by + cz) = af(x) + bf(y) + cf(z)
    """
    print(feature_vectors)
    n_feats, model_dimension = feature_vectors.shape
    n_chosen_feats = len(feature_ndxs)
    chosen_features = feature_vectors[feature_ndxs] # n_chosen_feats x model_dimension
    
    feature_activations = (torch.rand(n_samples, n_chosen_feats) * (feat_max - feat_min)) - feat_min # n_samples x n_chosen_feats
    together_vectors = feature_activations @ chosen_features # n_samples x n_chosen_feats @ n_chosen_feats x model_dimension = n_samples x model_dimension
    print(feature_activations.shape, together_vectors.shape, feature_activations.unsqueeze(-1).shape)
    single_feats = feature_activations.unsqueeze(-1) * torch.eye(n_chosen_feats) # n_samples x n_chosen_feats x n_chosen_feats
    single_vectors = single_feats.view(-1, n_chosen_feats) @ chosen_features # (n_samples x n_chosen_feats) x n_chosen_feats @ n_chosen_feats x model_dimension
    together_output = F.relu(together_vectors) # n_samples x model_dimension
    single_output = F.relu(single_vectors) # (n_samples x n_chosen_feats) x model_dimension
    single_sums = single_output.view(n_samples, n_chosen_feats, model_dimension).sum(dim=1) # n_samples x model_dimension
    # print(single_sums.shape, together_output.shape)
    # differences = together_output - single_sums
    # print(together_vectors, single_vectors)
    # print(together_output, single_sums, differences)
    
    # now we're going to define a linear and a non-linear model, and test the extent to which the non-linear model is better
    projected_outputs = together_output @ feature_vectors.T # n_samples x model_dimension @ model_dimension x n_feats = n_samples x n_feats 
    
    test_n = int(n_samples * test_frac)
    
    lin_reg = LinearRegression()
    lin_reg.fit(feature_activations[test_n:], projected_outputs[test_n:])
    # for the non-linear model, want to create an input dataset which contains all of the pairwise interactions between the features
    # so we want to create a dataset which is n_samples x n_chosen_feats ^2
    interactions_reg = LinearRegression()
    interactions = torch.zeros(n_samples, n_chosen_feats**2)
    for i in range(n_chosen_feats):
        for j in range(n_chosen_feats):
            interactions[:, i*n_chosen_feats + j] = feature_activations[:, i] * feature_activations[:, j]
    
    interactions_reg.fit(torch.cat([interactions, feature_activations], dim=1)[test_n:], projected_outputs[test_n:])
    
    lin_score = lin_reg.score(feature_activations[:test_n], projected_outputs[:test_n])
    interactions_score = interactions_reg.score(torch.cat([interactions, feature_activations], dim=1)[:test_n], projected_outputs[:test_n])
    
    # trying a different non-linearity metric where we define a direction in the space of the chosen features, and then look at the extent to which a quadratic model fits better than a linear model
    n_attempts = 100
    scores = []
    for i in range(n_attempts):
        feats_split = torch.rand(n_chosen_feats)
        direction = feats_split @ chosen_features # model_dimension
        direction /= direction.norm()
        dir_projections_in = together_vectors @ direction # n_samples
        dir_projections_out = together_output @ direction # n_samples
        dir_proj_sq = dir_projections_in**2
        lin_reg = LinearRegression()
        lin_reg.fit(dir_projections_in[test_n:].unsqueeze(1), dir_projections_out[test_n:])
        
        quad_reg = LinearRegression()
        print(dir_projections_out[test_n:].shape, torch.stack([dir_projections_in, dir_proj_sq], dim=1)[test_n:].shape)
        quad_reg.fit(torch.stack([dir_projections_in, dir_proj_sq], dim=1)[test_n:], dir_projections_out[test_n:])
        
        lin_score = lin_reg.score(dir_projections_in[:test_n].unsqueeze(1), dir_projections_out[:test_n])
        quad_score = quad_reg.score(torch.stack([dir_projections_in, dir_proj_sq], dim=1)[:test_n], dir_projections_out[:test_n])  
        
        scores.append(quad_score - lin_score)
              
    print(f"Average difference between linear and quadratic models: {np.mean(scores)}")
    
    return interactions_score - lin_score  # return average difference across all samples
    

def generate_dataset(
    n_samples,
    n_feats,
    feature_vectors,
    nonlin_fn,
    scale,
    bias,
    always_on_ndxs=None,
    sparsity=None,
    n_on=None,
):
    """Generate a sparse dataset with n_samples samples and n_feats features, where each feature is a linear combination of the feature vectors"""
    # can set sparsity
    if n_on is not None:
        if sparsity is not None:
            print(f"Warning: sparsity argument of {sparsity} is given but unused.")
        is_on_matrix = torch.zeros(n_samples, n_feats).to(torch.bool)
        for s in range(n_samples):
            on_ndxs = torch.randperm(n_feats)[:n_on]
            is_on_matrix[s, on_ndxs] = True
    else:
        assert sparsity is not None and sparsity > 0 and sparsity < 1, "Since n_on is not set, sparsity must be a float between 0.0 and 1.0"
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
    ax.scatter(
        vector_levels[:, feature_ndx],
        probed_input[:, feature_ndx],
        label="input",
        color="red",
    )
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


def measure_binary_nonhomogeneity(vector_levels, probed_input, probed_output, feature_ndx, test_prop=0.1):
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


def measure_nonhomogeneity(vector_levels, probed_output, feature_ndx, test_prop=0.9):
    assert vector_levels.shape == probed_output.shape
    n_samples = probed_output.shape[0]
    test_begin_ndx = int(n_samples * (1 - test_prop))
    quad_reg_inputs = torch.stack([vector_levels, vector_levels**2], dim=1)

    lin_reg = LinearRegression()
    lin_reg.fit(
        vector_levels[:test_begin_ndx, feature_ndx].unsqueeze_(1),
        probed_output[:test_begin_ndx, feature_ndx],
    )
    lin_reg_score = lin_reg.score(
        vector_levels[test_begin_ndx:, feature_ndx].unsqueeze_(1),
        probed_output[test_begin_ndx:, feature_ndx],
    )

    quad_reg = LinearRegression()
    quad_reg = quad_reg.fit(
        quad_reg_inputs[:test_begin_ndx, :, feature_ndx],
        probed_output[:test_begin_ndx, feature_ndx],
    )
    quad_reg_score = quad_reg.score(
        quad_reg_inputs[test_begin_ndx:, :, feature_ndx],
        probed_output[test_begin_ndx:, feature_ndx],
    )
    return quad_reg_score - lin_reg_score


def get_nonhomog_grid(n_samples, n_feats, n_on, feature_vectors, nonlin_fn, feature_ndx):
    nonhomogs = np.zeros((8, 10))
    for i, scale in enumerate([2**x for x in range(10)]):
        for j, bias in enumerate([-(2**x) for x in range(8)]):
            n_samples = 500
            inputs, outputs, vector_levels = generate_dataset(
                n_samples,
                n_feats,
                feature_vectors,
                nonlin_fn,
                scale,
                bias,
                always_on_ndxs=[feature_ndx],
                n_on=n_on,
            )
            probed_output = outputs @ feature_vectors.T
            nonhomogs[j, i] = measure_nonhomogeneity(vector_levels, probed_output, feature_ndx)
    
    return nonhomogs


def main():
    st.title("Streamlit for Neural Network Exploration")

    # Sidebar
    vector_dim = st.sidebar.slider("Vector Dimension", 5, 200, 100)
    dims_per_feature = st.sidebar.slider("Dims per Feature", 1, vector_dim, 3)
    n_feats = st.sidebar.slider("Number of Features", 50, 2000, 100)
    n_on = st.sidebar.slider("Number of other active features", 0, n_feats, 5, 1)
    nonlin = st.sidebar.selectbox("Nonlinearity", ["relu", "gelu", "tanh"])
    bias = st.sidebar.slider("Bias", -10.0, 10.0, 0.0, 0.1)
    scale = st.sidebar.slider("Scale", 0.5, 50.0, 5.0, 0.5)
    feature_ndx = st.sidebar.slider("Feature Index for Plotting", 0, n_feats - 1, 0)

    nonlin_dict = {
        "relu": torch.nn.ReLU(),
        "gelu": torch.nn.GELU(),
        "tanh": torch.nn.Tanh(),
    }
    nonlin_options = ["relu", "gelu", "tanh"]
    nonlin = st.sidebar.selectbox("Select Nonlinearity:", nonlin_options)
    nonlin_fn = nonlin_dict[nonlin]
    
    logger.info(f"Running page with: Vector Dimension: {vector_dim} | Dims per Feature: " + \
        f"{dims_per_feature} | Number of Features: {n_feats} | Number of active features: {n_on} " + \
        f"| Nonlinearity: {nonlin} | Bias: {bias} | Scale: {scale} | Feature Index for Plotting: {feature_ndx}")

    feature_vectors = compute_feature_vectors(vector_dim, dims_per_feature, n_feats)

    # Main Content
    st.write("## Dataset Generation")
    n_samples = 5000
    inputs, outputs, vector_levels = generate_dataset(
        n_samples,
        n_feats,
        feature_vectors,
        nonlin_fn,
        scale,
        bias,
        always_on_ndxs=[feature_ndx],
        n_on=n_on,
    )

    # Plotting the input output relationship as a scatter plot
    st.write("## Response Curves")
    probed_input = inputs @ feature_vectors.T
    probed_output = outputs @ feature_vectors.T
    fig = plot_response_curves(probed_input, probed_output, vector_levels, feature_ndx)
    st.pyplot(fig)

    # Plotting the non-homogeneity scores
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    n_samples = 500  # lower since we're doing a large number of tests

    nonhomogs = get_nonhomog_grid(n_samples, n_feats, n_on, feature_vectors, nonlin_fn, feature_ndx)
    cax = ax.imshow(nonhomogs, vmin=0, vmax=0.2)
    ax.set_xlabel("Feature input scale")
    ax.set_ylabel("MLP Bias")
    ax.set_title("Non-homogeneity measure by feature scale and mlp bias.")
    ax.set_xticks(list(range(10)))
    ax.set_yticks(list(range(8)))
    ax.set_xticklabels([2**x for x in range(10)])
    ax.set_yticklabels([-(2**x) for x in range(8)])
    fig.colorbar(cax, shrink=0.3, aspect=15 * 0.5)
    st.pyplot(plt, use_container_width=True)

    st.write(f"Max non-homogeneity score: {np.max(nonhomogs):.2}")
    
    # Now we test linearity 
    n_feats = 5
    lin_test = test_linearity(feature_vectors=feature_vectors, feature_ndxs=list(range(n_feats)), n_samples = 500)
    print(lin_test)
    st.write(f"Linearity test score: {lin_test:.10}")
    


if __name__ == "__main__":
    main()