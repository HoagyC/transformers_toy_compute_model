# %%
import torch
import numpy as np
from matplotlib import pyplot as plt

# %%
nonlin_dict = {
    "relu": torch.nn.ReLU(),
    "gelu": torch.nn.GELU(),
    "tanh": torch.nn.Tanh(),
}

vector_dim = 100
dims_per_feature = 30


n_feats = 100
sparsity = 0.9
nonlin = "relu"
nonlin_fn = nonlin_dict[nonlin]
bias = 0

assert dims_per_feature < vector_dim

feature_vectors = torch.rand(n_feats, vector_dim) - 0.5

# zero out all but a random set of dims_per_feature dimensions
for i in range(n_feats):
    dead_feats = np.random.choice(vector_dim, vector_dim - dims_per_feature, replace=False)
    feature_vectors[i, dead_feats] = 0
    feature_vectors[i] /= feature_vectors[i].norm()
    assert feature_vectors[i].norm() - 1 < 1e-5

# generate sparse dataset
n_samples = 5000

is_on_matrix = torch.rand(n_samples, n_feats) > sparsity

negative_vector_levels = torch.zeros(n_samples, n_feats)
positive_vector_levels = torch.rand(n_samples, n_feats) * 50
vector_levels = torch.where(is_on_matrix, positive_vector_levels, negative_vector_levels)

# vector_levels = torch.rand(n_samples, n_feats)

# generate input vectors
inputs = vector_levels @ feature_vectors  # n_samples x vector_dim
outputs = nonlin_fn(inputs + bias)  # n_samples x vector_dim

# check the response curve of the output and input when we probe for a single feature with known direction
probed_input = inputs @ feature_vectors.T  # n_samples x n_feats
probed_output = outputs @ feature_vectors.T  # n_samples x n_feats

# plot the response curves
feature_ndx = 0
plt.figure()
plt.scatter(
    vector_levels[:, feature_ndx],
    probed_input[:, feature_ndx],
    label="input",
    color="red",
)
plt.scatter(vector_levels[:, feature_ndx], probed_output[:, feature_ndx], label="output")
plt.title("Response curve for a single feature")
plt.legend()
plt.xlabel("input level")
plt.ylabel("output level")
plt.show()

# now take a running average and plot the response curves
window_size = 50
probed_input_avg = torch.zeros(n_samples)
probed_output_avg = torch.zeros(n_samples)

# sort the probed_input bty the ordering of the vector_levels
ndxs = torch.argsort(vector_levels[:, feature_ndx])
probed_input_sorted = probed_input[ndxs, feature_ndx]
probed_output_sorted = probed_output[ndxs, feature_ndx]

for i in range(n_samples):
    probed_input_avg[i] = probed_input_sorted[max(0, i - window_size) : min(n_samples, i + window_size)].mean(dim=0)
    probed_output_avg[i] = probed_output_sorted[max(0, i - window_size) : i + 1].mean(dim=0)


plt.figure()
plt.scatter(vector_levels[ndxs, feature_ndx], probed_input_avg[:], label="input", color="red")
plt.scatter(vector_levels[ndxs, feature_ndx], probed_output_avg[:], label="output")
plt.title("Moving average of response curves")
plt.legend()

plt.xlabel("input level")
plt.ylabel("output level")
plt.show()


# need a metric of linearity, and a metric of degree of separation
# the metric of linearity is going to be the degree to which a quadratic, or other similar non-linear function, can fit the data
# relative to a linear function

# linear classifier for probed_output vs vector_levels
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(vector_levels, probed_output)
lin_reg_score = lin_reg.score(vector_levels, probed_output)

quad_reg = LinearRegression()
quad_reg_inputs = torch.cat([vector_levels, vector_levels**2], dim=1)
quad_reg = quad_reg.fit(quad_reg_inputs, probed_output)
quad_reg_score = quad_reg.score(quad_reg_inputs, probed_output)

print(f"Linear regression score: {lin_reg_score:.3f}")
print(f"Quadratic regression score: {quad_reg_score:.3f}")
print(f"Difference: {quad_reg_score - lin_reg_score:.3f}")


# %%
def single_run(
    vector_dim: int = 100,
    dims_per_feature: int = 30,
    n_feats: int = 10,
    sparsity: float = 0.9,
    nonlin: str = "relu",
    bias: float = -0.5,
    n_regs=30,
    test_prop=0.1,
):
    nonlin_fn = nonlin_dict[nonlin]
    assert dims_per_feature < vector_dim

    feature_vectors = torch.rand(n_feats, vector_dim)

    # zero out all but a random set of dims_per_feature dimensions
    for i in range(n_feats):
        dead_feats = np.random.choice(vector_dim, vector_dim - dims_per_feature, replace=False)
        feature_vectors[i, dead_feats] = 0
        feature_vectors[i] /= feature_vectors[i].norm()
        assert feature_vectors[i].norm() - 1 < 1e-5

    # generate sparse dataset
    n_samples = 50000

    is_on_matrix = torch.rand(n_samples, n_feats) > sparsity

    negative_vector_levels = torch.zeros(n_samples, n_feats)
    positive_vector_levels = torch.rand(n_samples, n_feats)
    vector_levels = torch.where(is_on_matrix, positive_vector_levels, negative_vector_levels)

    # vector_levels = torch.rand(n_samples, n_feats) ** 3

    # vector_levels = torch.rand(n_samples, n_feats)

    # generate input vectors
    inputs = vector_levels @ feature_vectors  # n_samples x vector_dim
    outputs = nonlin_fn(inputs + bias)  # n_samples x vector_dim

    # check the response curve of the output and input when we probe for a single feature with known direction
    probed_input = inputs @ feature_vectors.T  # n_samples x n_feats
    probed_output = outputs @ feature_vectors.T  # n_samples x n_feats

    test_begin_ndx = int(n_samples * (1 - test_prop))
    quad_reg_inputs = torch.stack([vector_levels, vector_levels**2], dim=1)

    differences = np.empty(min(n_regs, n_feats))
    for feature_ndx in range(min(n_regs, n_feats)):
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
        differences[feature_ndx] = quad_reg_score - lin_reg_score
        if abs(quad_reg_score - lin_reg_score) > 1:
            print(quad_reg_score, lin_reg_score)

    return differences.mean()


# %%
sparsity = 0.95
differences = np.empty((5, 6))
for i, n_feats in enumerate([2**x for x in range(5, 10)]):
    for j, dims_per_feature in enumerate([2**x for x in range(0, 6)]):
        differences[i, j] = single_run(n_feats=n_feats, dims_per_feature=dims_per_feature, sparsity=sparsity)


# %%
# display the results, with dims_per_feature on the x axis, and n_feats on the y axis, noting that it can be negative so cant sue imshow
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(differences.T, cmap="RdBu", vmin=np.min(differences), vmax=np.max(differences))
fig.colorbar(im, ax=ax)
ax.set_xticklabels([0] + [2**x for x in range(5, 10)])
ax.set_yticklabels([0] + [2**x for x in range(0, 6)])

ax.set_xlabel("n_feats")
ax.set_ylabel("dims_per_feature")
ax.set_title(f"Non-linearity metric in a 100 dim space at {sparsity} sparsity")


# %%
# constructing an interesting distributed representation
width = 1000
input_direction = torch.ones(width) / np.sqrt(width)
all_inputs = torch.outer(torch.range(-50, 10, 0.1), input_direction)
noise = torch.randn_like(all_inputs)
all_inputs += noise
all_outputs = torch.nn.GELU()(all_inputs)
projected_outputs = all_outputs @ input_direction
projected_inputs = all_inputs @ input_direction
# %%
plt.scatter(projected_inputs, projected_outputs)
# %%
