import logging
from typing import List, Tuple

import streamlit as st
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from transformer_lens import HookedTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@st.cache_data
def compute_feature_vectors(vector_dim: int, dims_per_feature: int, n_feats: int, include_negative: bool = True):
    """Generate a set of feature vectors, each of which is a unit vector with dims_per_feature non-zero dimensions"""
    feature_vectors = torch.rand(n_feats, vector_dim)
    if include_negative:
        feature_vectors -= 0.5
    for i in range(n_feats):
        dead_feats = np.random.choice(vector_dim, vector_dim - dims_per_feature, replace=False)
        feature_vectors[i, dead_feats] = 0
        feature_vectors[i] /= feature_vectors[i].norm()
    return feature_vectors

@st.cache_data
def compute_soft_aligned_features(vector_dim: int, n_feats: int, pow: float, include_negative: bool = True):
    feature_vectors = torch.rand(n_feats, vector_dim)
    feature_vectors = feature_vectors ** pow
    feature_vectors /= feature_vectors.norm(dim=1).unsqueeze(1)
    if include_negative:
        feature_vectors *= (torch.randint_like(feature_vectors, 2) * 2) - 1
    
    return feature_vectors.squeeze(1)

def check_most_similar(feature_vectors):
    """
    Want to find the average level of similarity 
    """
    n_feats, vector_dim = feature_vectors.shape
    assert torch.allclose(feature_vectors.norm(dim=1), torch.ones(n_feats))
    cos_sim_matrix = torch.einsum("ac, bc -> ab", [feature_vectors, feature_vectors])
    cos_sim_matrix[torch.eye(n_feats).bool()] = 0 # blank out the diagonal
    max_cos_sims, _ = torch.max(cos_sim_matrix, dim=1) # blank is indices
    return max_cos_sims.mean()
    

def test_linearity(feature_vectors, feature_ndxs: List[int], n_samples: int, feat_min: int = 0, feat_max: int = 5, test_frac: float = 0.2):
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
    biases,
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
    positive_vector_levels = (torch.rand(n_samples, n_feats) - 0.5) * (scale * 2)
    vector_levels = torch.where(is_on_matrix, positive_vector_levels, negative_vector_levels)
    inputs = vector_levels @ feature_vectors
    assert biases.shape == inputs.shape[1:], f"Biases must have shape {inputs.shape[1:]}. Got {biases.shape} instead."
    outputs = nonlin_fn(inputs + biases)
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


def measure_binary_nonhomogeneity(
    vector_levels, probed_input, probed_output, feature_ndx, test_prop=0.1):
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

def write_2d_and_gate():
    """
    Streamlit section which constructs a two neuron system which can function as an OR gate when
    the output is viewed in the correct basis (due to Jake Mendel). 
    
    Can use sliders to view the input-output behaviour in terms of the angle of the directions 
    written to and read from.
    """
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    n1 = np.maximum(X + Y, 0)
    n2 = np.maximum(X + Y - 1, 0)
    # show n1 and n2 as image
    # fig, ax = plt.subplots()
    # ax.imshow(n1, cmap="gray")
    # st.pyplot(fig)
    # ax.imshow(n2, cmap="gray")
    # st.pyplot(fig)
    # now we can start to ask, which directions are interesting?
    input_theta = st.slider("Input Theta", 0.0, 2 * np.pi, 0.0, 0.02)
    output_theta = st.slider("Output Theta", 0.0, 2 * np.pi, -np.pi / 4, 0.02)
    # output_theta = input_theta
    output_theta = -np.pi / 4
    input_line = np.linspace(-2, 2, 100)
    # now want 100 points in x,y space along the line with theta
    x = np.cos(input_theta) * input_line
    y = np.sin(input_theta) * input_line
    input_vals = np.stack([x, y], axis=1)
    n1 = np.maximum(input_vals[:, 0] + input_vals[:, 1], 0.)
    n2 = np.maximum(input_vals[:, 0] + input_vals[:, 1] - 1, 0.)
    output_vals = np.stack([n1, n2], axis=1)
    # now we project onto the output direction
    output_vals = output_vals @ np.array([np.cos(output_theta), np.sin(output_theta)])
    plt.clf()
    plt.plot(input_line, output_vals)
    st.pyplot()
    


def write_sawtooth_maker():
    # Ok this time we're going to construct a deliberately cool pattern, a saw tooth that goes on for n_neurons / 2 teeth
    x = torch.linspace(0., 10, 1000)
    noise_scale = st.slider("Noise Scale", 0.0, 1.0, 0.0, 0.01)
    noisy_x = x + torch.randn_like(x) * noise_scale
    n1 = torch.nn.ReLU()(noisy_x - 1)
    n2 = torch.nn.ReLU()(noisy_x - 2)     
    n3 = torch.nn.ReLU()(noisy_x - 3)     
    n4 = torch.nn.ReLU()(noisy_x - 4)     
    n5 = torch.nn.ReLU()(noisy_x - 5)
    output = n1 - 2*n2 + 2*n3 - 2*n4 + 2*n5
    plt.clf()
    plt.scatter(x, output)
    st.write("saw tooth")
    st.pyplot()


def write_output_predictor():
    """
    Here we test the degree to which we can reconstruct random directions which are more or less 
    basis-aligned from the inputs. 
    
    The outcome is that the quality of reconstruction is consistent across different levels of basis
    alignment, showing that it's not abou
    """
    n_dimensions = 10
    n_samples = 1000
    test_frac = 0.2
    test_n = int(n_samples * test_frac)
    random_input = torch.randn(n_samples, n_dimensions)
    norm_power = st.slider("Norm power", 0.5, 20., 1.0, 0.1)
    include_negative = st.checkbox("Use negatives?")
    random_vec = torch.rand(n_dimensions)
    random_vec = (random_vec ** norm_power)
    if include_negative:
        random_vec *= (torch.randint(0, 2, (n_dimensions,)) * 2 - 1) 
    
    outputs = F.relu(random_input)
    direction_outputs = outputs @ random_vec
    lin_reg = LinearRegression()
    lin_reg.fit(random_input[-test_n:], direction_outputs[-test_n:])
    lin_score = lin_reg.score(random_input[:-test_n], direction_outputs[:-test_n])
    st.write(f"Direction is {random_vec}, Linearity score is {lin_score}")
    
    
def write_old_intro():
    st.title("Streamlit for Neural Network Exploration")
    
    st.write("The question I want to understand is how we should undertstand the relationship between feature formation and the nonlinearity of the neural network.")
    st.write("In particular, I want to ask whether the formation of features needs to be in some way basis-aligned with the non-linearity.")
    
    st.write("Let's first look at a single neuron, using the ReLU and GELU non-linearities")
    
    # Draw a plot of the response curve for a single feature
    st.write("The way I'm going to look at neurons is through plotting 'response curves', where I vary the input along a particular direction, and plot the level of output along that direction.")
    st.write("I will also plot the *input* in that direction. When there's no noise in the input, this is just the level of the feature, but as we start to add noise, it will be the level of the feature plus some noise.")
    
    # single_relu
    inputs, outputs, vector_levels = generate_dataset(
        n_samples=1000,
        n_feats=1,
        feature_vectors=torch.tensor([[1.0]]),
        nonlin_fn=torch.nn.GELU(),
        scale=5.0,
        biases=torch.tensor([-2.5]),
        always_on_ndxs=[0],
        n_on=0,
    )
    fig = plot_response_curves(inputs, outputs, vector_levels, feature_ndx=0)
    st.pyplot(fig)
    
    # We can compose the action of multiple non-linearities to make more complex functions
    feature_vec = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    inputs, outputs, vector_levels = generate_dataset(
        n_samples=1000,
        n_feats=2,
        feature_vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        nonlin_fn=torch.nn.ReLU(),
        scale=5.0,
        biases=torch.tensor([-2., -4.]),
        always_on_ndxs=[0, 1],
        n_on=0,
    )
    probed_input = inputs @ feature_vec
    probed_output = outputs @ feature_vec
    fig = plot_response_curves(probed_input, probed_output, vector_levels, feature_ndx=1)
    st.pyplot(fig)

def write_input_output_norm():
    dim = st.slider("Dimension", 1, 100, 10)
    norm_power = st.slider("Norm power", 0.5, 20., 1.0, 0.1)
    logger.info(f"Running with dim {dim} and norm power {norm_power}")
    random_vec = torch.rand(dim)
    
    random_vec = (random_vec ** norm_power) * (torch.randint(0, 2, (dim,)) * 2 - 1) 
    print(f"Random vec norm {torch.norm(random_vec)}, vec {random_vec}")
    random_vec = random_vec / torch.norm(random_vec)
    input = torch.randn(1000, dim)
    bias_scale = 5.
    relu_output = torch.nn.ReLU()(input + torch.rand(dim) * bias_scale - bias_scale / 2)
    line_output = relu_output @ random_vec
    line_input = input @ random_vec
    plt.clf()
    plt.scatter(line_input, line_output)
    print(line_input, line_output)
    st.pyplot()
    
    
@st.cache_data
def make_nonhomog_grid(vector_dim: int, n_feats: int, bias_scale: float, nonlin_fn_str: str = "gelu", n_feats_tested: int = 10, n_samples: int = 5000):
    nonlin_dict = {"relu": torch.nn.ReLU(), "gelu": torch.nn.GELU()}
    nonlin_fn = nonlin_dict[nonlin_fn_str]
    scale = np.sqrt(vector_dim)
    n_on_l = sorted(list(set([0] + [int(2 ** x)for x in np.linspace(0, np.log2(vector_dim), 20)])))
    dims_per_feature_l = sorted(list(set([int(2 ** x) for x in np.linspace(1, np.log2(vector_dim), 20)] + [vector_dim])))
    nonhomog_grid = np.zeros((len(dims_per_feature_l), len(n_on_l)))
    for i, dims_per_feature in enumerate(dims_per_feature_l): 
        for j, n_on in enumerate(n_on_l):       
            feature_vectors = compute_feature_vectors(vector_dim, dims_per_feature, n_feats, include_negative=False)
            biases = torch.rand(vector_dim) * bias_scale
            nonhomog_scores = []
            for feature_ndx in range(n_feats_tested):
                _inputs, outputs, vector_levels = generate_dataset(
                    n_samples,
                    n_feats,
                    feature_vectors,
                    nonlin_fn,
                    scale,
                    biases,
                    always_on_ndxs=[feature_ndx],
                    n_on=n_on,
                )
                probed_output = outputs @ feature_vectors.T
                nonhomog_score = measure_nonhomogeneity(vector_levels, probed_output, feature_ndx)
                nonhomog_scores.append(nonhomog_score)
            
            nonhomog_grid[i, j] = np.mean(nonhomog_scores)
        
    # add rows and columns to the grid
    nonhomog_grid = pd.DataFrame(nonhomog_grid)
    nonhomog_grid.columns = n_on_l
    nonhomog_grid.index = dims_per_feature_l
    # # add label to the index and columns
    # nonhomog_grid.index = pd.MultiIndex.from_product([['Dims per feature'], nonhomog_grid.index])
    # nonhomog_grid.columns = pd.MultiIndex.from_product([['N on'], nonhomog_grid.columns])
    return nonhomog_grid

@st.cache_data
def make_interference_grid(vector_dim: int, n_tries: int = 1):
    dims_per_features = sorted(list(set([int(2 ** x) for x in np.linspace(1, np.log2(vector_dim), 20)] + [vector_dim])))
    n_feats_l = sorted(list(set([int(2 ** x)for x in np.linspace(4, 14, 20)])))
    table = np.zeros((len(dims_per_features), len(n_feats_l)))
    for i, dims_per_feature in enumerate(dims_per_features):
        for j, n_feats in enumerate(n_feats_l):
            mean_maxes = []
            for _ in range(n_tries):
                feature_vectors = compute_feature_vectors(vector_dim, dims_per_feature, n_feats)
                mean_max_cos = check_most_similar(feature_vectors)
                mean_maxes.append(mean_max_cos)
            table[i, j] = np.mean(mean_maxes)
    
    # add rows and columns to the grid
    table = pd.DataFrame(table)
    table.columns = n_feats_l
    table.index = dims_per_features
    # # add label to the index and columns
    # table.columns = pd.MultiIndex.from_product([['Number of features'], table.columns])
    # table.index = pd.MultiIndex.from_product([['Dims per feature'], table.index])
    return table
                

def write_homogeneity_testing():
    # Sidebar
    vector_dim = st.sidebar.slider("Vector Dimension", 5, 200, 100)
    dims_per_feature = st.sidebar.slider("Dims per Feature", 1, vector_dim, 3)
    n_feats = st.sidebar.slider("Number of Features", 50, 2000, 100)
    n_on = st.sidebar.slider("Number of other active features", 0, n_feats, 0, 1)
    bias_scale = st.sidebar.slider("Bias", -10.0, 10.0, 0.0, 0.1)
    scale = np.sqrt(dims_per_feature) * 2
    # scale = st.sidebar.slider("Scale", 0.5, 50.0, 5.0, 0.5)
    feature_ndx = st.sidebar.slider("Feature Index for Plotting", 0, n_feats - 1, 0)
    include_negative = st.sidebar.checkbox("Include negative features", False)

    nonlin_dict = {
        "relu": torch.nn.ReLU(),
        "gelu": torch.nn.GELU(),
        "tanh": torch.nn.Tanh(),
    }
    nonlin_options = ["relu", "gelu", "tanh"]
    nonlin = st.sidebar.selectbox("Select Nonlinearity:", nonlin_options, 1)
    nonlin_fn = nonlin_dict[nonlin]
    
    logger.info(f"Running page with: Vector Dimension: {vector_dim} | Dims per Feature: " + \
        f"{dims_per_feature} | Number of Features: {n_feats} | Number of active features: {n_on} " + \
        f"| Nonlinearity: {nonlin} | Bias scale: {bias_scale} | Scale: {scale} | Feature Index for Plotting: {feature_ndx}")

    feature_vectors = compute_feature_vectors(vector_dim, dims_per_feature, n_feats, include_negative)

    # Main Content
    st.write("## Dataset Generation")
    n_samples = 5000
    biases = torch.rand(vector_dim) * bias_scale
    inputs, outputs, vector_levels = generate_dataset(
        n_samples,
        n_feats,
        feature_vectors,
        nonlin_fn,
        scale,
        biases,
        always_on_ndxs=[feature_ndx],
        n_on=n_on,
    )

    # Plotting the input output relationship as a scatter plot
    st.write("## Response Curves")
    probed_input = inputs @ feature_vectors.T
    probed_output = outputs @ feature_vectors.T
    fig = plot_response_curves(probed_input, probed_output, vector_levels, feature_ndx)
    st.pyplot(fig)

    # nonhomog_score = measure_nonhomogeneity(vector_levels, probed_output, feature_ndx)
    # st.write(f"## Nonhomogeneity Score: {nonhomog_score:.10}")
    
    nonhomog_grid = make_nonhomog_grid(vector_dim, n_feats, bias_scale, nonlin)
    st.write("## Nonhomogeneity Grid")
    st.table(nonhomog_grid)
    
    interference_table = make_interference_grid(vector_dim)
    st.table(interference_table)
    
    # ok from this table I now want to get, for each row (dims per feature) the number of features that are need to get a homogeneity score above 0.1
    # and also to get a level of interference below 0.25.
    min_homog = 0.1
    max_interference = 0.30
    
    most_feats_while_nonhomog = []
    most_feats_without_interfering = []
    
    for _, row in interference_table.iterrows():
        n_feats_while_interfering = 0
        for j, val in enumerate(row):
            if val > max_interference:
                n_feats_while_interfering = interference_table.columns[j]
                break
        most_feats_without_interfering.append(n_feats_while_interfering)
        
    for _, row in nonhomog_grid.iterrows():
        n_feats_while_homog = nonhomog_grid.columns[-1]
        for j, val in enumerate(row[::-1]):
            if val > min_homog:
                n_feats_while_homog = nonhomog_grid.columns[-(j+1)]
                break
        most_feats_while_nonhomog.append(n_feats_while_homog)
        
    st.write("## Number of Features for Homogeneity and Interference")
    st.write("The number of features required to get a homogeneity score above 0.1 and an interference score below 0.25")
    st.write("### Homogeneity")
    st.write(most_feats_while_nonhomog)
    st.write("### Interference")
    st.write(most_feats_without_interfering)
    
    # plot the number of features required to get a homogeneity score above 0.1 and an interference score below 0.3 
    # as a function of the number of dimensions per feature, using separate axes on the left and right of the plot.
    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_left.plot(nonhomog_grid.index, most_feats_while_nonhomog, color="blue")
    ax_left.set_xlabel("Dimensions per Feature")
    ax_left.set_ylabel("Maximum active feats without becoming homogeneous", color="blue")
    ax_right = ax_left.twinx()
    ax_right.plot(interference_table.index, most_feats_without_interfering, color="red")
    ax_right.set_ylabel("Maximum number of features with only limited interefence", color="red")
    # set all axes to be log scale
    ax_left.set_xscale("log")
    ax_right.set_yscale("log")
    ax_left.set_yscale("log")
    ax_left.set_title("Tradeoff between non-homogeneity and interference as dims per feat increases (vector_dim=100)")
    ax_left.xaxis.set_major_formatter(ScalarFormatter())
    ax_left.xaxis.set_minor_formatter(NullFormatter())
    ax_left.yaxis.set_major_formatter(ScalarFormatter())
    ax_left.yaxis.set_minor_formatter(NullFormatter())
    ax_right.yaxis.set_major_formatter(ScalarFormatter())
    ax_right.yaxis.set_minor_formatter(NullFormatter())
    
    
    st.pyplot(fig)

    

def write_linearity_test():
    """
    Testing whether there's a linear relationship between the input when expressed in terms of 
    the levels of features spread across many dimensions, and the output in a particular direction.
    
    This generally comes up negative. Annoyingly the question is not 'is this a neat linear function of the input' (because it is not)
    but whether there exists some interesting function in the 
    """
    st.write("Testing whether there's a linear relationship between the input when expressed in terms of " + \
         "the levels of features spread across many dimensions, and the output in a particular direction.")
    dimension = st.slider("Dimension", 1, 100, 10)
    n_feats = st.slider("Number of Features", 1, 500, 10)
    input_pow = st.slider("Input Power", 0.5, 20., 1.0, 0.1)
    output_pow = st.slider("Output Power", 0.5, 20., 1.0, 0.1)
    n_on = st.slider("Number of active features", 0, n_feats, 5, 1)
    feats = compute_soft_aligned_features(dimension, n_feats, input_pow)
    measure_dim = compute_soft_aligned_features(dimension, 1, output_pow)
    n_samples = 1000
    test_frac = 0.2
    test_n = int(n_samples * test_frac)
    inputs, outputs, vector_levels = generate_dataset(
        n_samples=n_samples,
        n_feats = n_feats,
        feature_vectors=feats,
        nonlin_fn=torch.nn.ReLU(),
        scale = 5.,
        biases = torch.rand(dimension) * 5.,
        n_on=n_on
    )
    
    lin_reg = LinearRegression()
    dimension_outputs = outputs @ measure_dim.T
    lin_reg.fit(vector_levels[:-test_n], dimension_outputs[:-test_n])
    score = lin_reg.score(vector_levels[-test_n:], dimension_outputs[-test_n:])
    st.write(f"Linear regression score: {score:.2}")
    
def write_lw_100_example():
    # First writing the case with a aligned feature
    plt.clf()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    input_vals = torch.linspace(-5, 5, 1000)
    aligned_feat = torch.zeros(100)
    aligned_feat[0] = 1
    distributed_feat = torch.ones(100)
    distributed_feat /= distributed_feat.norm()

    aligned_input_vals = torch.outer(input_vals, aligned_feat)
    distributed_input_vals = torch.outer(input_vals, distributed_feat)
    
    aligned_input_postnonlin = F.gelu(aligned_input_vals)
    distributed_input_postnonlin = F.gelu(distributed_input_vals)
    
    aligned_in_aligned_out = aligned_input_postnonlin @ aligned_feat
    aligned_in_distributed_out = aligned_input_postnonlin @ distributed_feat
    distributed_in_aligned_out = distributed_input_postnonlin @ aligned_feat
    distributed_in_distributed_out = distributed_input_postnonlin @ distributed_feat
    
    axs[0][0].plot(input_vals, aligned_in_aligned_out)
    axs[0][0].set_title("aligned input, aligned output")
    axs[0][1].plot(input_vals, aligned_in_distributed_out)
    axs[0][1].set_title("aligned input, distributed output")
    axs[1][0].plot(input_vals, distributed_in_aligned_out)
    axs[1][0].set_title("distributed input, aligned output")
    axs[1][1].plot(input_vals, distributed_in_distributed_out)
    axs[1][1].set_title("distributed input, distributed output")

    st.write(aligned_in_aligned_out.min(), distributed_in_distributed_out.min())
    
    st.pyplot(plt, use_container_width=True)
    
    plt.clf()
    small_input_range = torch.linspace(-5, 5, 100)
    large_input_range = small_input_range * 10
    small_aligned_input_vals = torch.outer(small_input_range, aligned_feat)
    large_distributed_input_vals = torch.outer(large_input_range, distributed_feat)
    aligned_in_aligned_out_small = F.gelu(small_aligned_input_vals) @ aligned_feat
    distributed_in_distributed_out_large = F.gelu(large_distributed_input_vals) @ distributed_feat
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(small_input_range, aligned_in_aligned_out_small)
    axs[0].set_title("aligned input, aligned output, small range")
    axs[1].plot(large_input_range, distributed_in_distributed_out_large)
    axs[1].set_title("distributed input, distributed output, large range")
    
    st.pyplot(plt, use_container_width=True)
    
    # Now I will show what happens when the elements of the distributed feature vector are a mixture of positive and negative
    plt.clf()
    input_vals = large_input_range
    n_negs = [0, 10, 25, 40, 50, 75, 90, 100]
    
    n_graph_rows = (len(n_negs) // 4) + (len(n_negs) % 4 > 0)
    fig, axs = plt.subplots(n_graph_rows, min(4, len(n_negs)), figsize=(12, 4 * n_graph_rows))
    for i, n_neg in enumerate(n_negs):
        distributed_feat = torch.ones(100)
        distributed_feat[:n_neg] = -1
        distributed_feat /= distributed_feat.norm()
        distributed_input_vals = torch.outer(input_vals, distributed_feat)
        distributed_input_postnonlin = F.gelu(distributed_input_vals)
        distributed_output = distributed_input_postnonlin @ distributed_feat
        axs[i // 4][i % 4].plot(input_vals, distributed_output)
        axs[i // 4][i % 4].set_title(f"{n_neg} negative elements")
    
    st.pyplot(plt, use_container_width=True)
    
    # Ok now we're going to look at how two different features interact
    feat_a = torch.zeros(100)
    feat_a[:50] = 1
    feat_a /= feat_a.norm()
    
    feat_b = torch.ones(100)
    # feat_b[:25] = 1
    # feat_b[75:] = 1
    feat_b /= feat_b.norm()
    
    # looking at the interaction between the two features, in the output space of the first feature
    input_range = small_input_range * np.sqrt(50) # use the sqrt of the number of elements to keep the scale of the non-linearity constant
    input_grid = torch.stack(torch.meshgrid(input_range, input_range), dim=-1)
    input_vecs = input_grid.reshape(-1, 2) @ torch.stack([feat_a, feat_b], dim=0)
    output_vecs = F.gelu(input_vecs)
    output_feats = (output_vecs @ feat_a).reshape(100, 100)
    plt.clf()
    plt.imshow(output_feats)
    plt.colorbar()
    st.pyplot(plt, use_container_width=True)

    # Now repeating the experiment, but with only 2 dimensions
    feat_a = torch.zeros(2)
    feat_a[0] = 1
    
    feat_b = torch.ones(2)
    feat_b /= feat_b.norm()
    
    input_range = small_input_range * np.sqrt(2) # use the sqrt of the number of elements to keep the scale of the non-linearity constant
    input_grid = torch.stack(torch.meshgrid(input_range, input_range), dim=-1)
    input_vecs = input_grid.reshape(-1, 2) @ torch.stack([feat_a, feat_b], dim=0)
    output_vecs = F.gelu(input_vecs)
    output_feats = (output_vecs @ feat_a).reshape(100, 100)
    plt.clf()
    plt.imshow(output_feats)
    plt.colorbar()
    st.pyplot(plt, use_container_width=True)
    
    # Now testing the case where we have two features with 50 positive and 50 negative elements, to see if there's interesting interactions
    feat_a = torch.ones(100)
    neg_feats = torch.randperm(100)[:50]
    feat_a[neg_feats] = -1
    feat_a /= feat_a.norm()
    
    feat_b = torch.ones(100)
    neg_feats = torch.randperm(100)[:50]
    feat_b[neg_feats] = -1
    feat_b /= feat_b.norm()
    
    st.write(f"Experimenting with 50 positive elements and 50 negative elements randomly across 2 features")
    input_range = small_input_range * np.sqrt(100) # use the sqrt of the number of elements to keep the scale of the non-linearity constant
    input_grid = torch.stack(torch.meshgrid(input_range, input_range), dim=-1)
    input_vecs = input_grid.reshape(-1, 2) @ torch.stack([feat_a, feat_b], dim=0)
    output_vecs = F.gelu(input_vecs)
    output_feats = (output_vecs @ feat_a).reshape(100, 100)
    mixed_feature = feat_a + feat_b
    mixed_feature /= mixed_feature.norm()
    mixed_feats = (output_vecs @ mixed_feature).reshape(100, 100)
    plt.clf()
    plt.imshow(output_feats)
    plt.title("Output of feature A")
    plt.colorbar()
    st.pyplot(plt, use_container_width=True)
    
    plt.clf()
    plt.imshow(mixed_feats)
    plt.title("Output of Feature A + Feature B")
    plt.colorbar()
    st.pyplot(plt, use_container_width=True)
    
    
def load_sae_features():
    load_location = "inputs/untied_mlp_l2_r2_e59/learned_dicts.pt"
    import sys
    sys.path.append("../sparse_coding")
    sae_list = torch.load(load_location, map_location=torch.device('cpu'))
    sae = sae_list[4][0]
    features = sae.get_learned_dict() # n_features x model_dim (4096 x 2048)
    return features 


def get_pythia_data(batch_size: int = 256, layer = 2, layer_loc: str = "mlp"):
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device="cpu")
    line_dataset = load_dataset("NeelNanda/pile-10k")
    dataloader = DataLoader(
        line_dataset,
        batch_size=batch_size,
    )

def write_all_lw_graphs():
    write_lw_100_example()
    write_homogeneity_testing()
    
def main():
    write_all_lw_graphs()



if __name__ == "__main__":
    main()
