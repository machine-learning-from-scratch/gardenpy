import numpy as np

lr = 0.01


def sgd_backward(nodes, expected, weights, biases):
    # initialize gradient lists
    d_weights = []
    d_biases = []

    d_b = -2 * (expected - nodes[-1])
    d_biases.insert(0, d_b)
    for layer in range(-1, -len(nodes) + 1, -1):
        d_w = nodes[layer - 1].T * d_b
        d_weights.insert(0, d_w)
        d_b = np.array([np.sum(weights[layer] * d_b, axis=1)])
        d_biases.insert(0, d_b)
    d_w = nodes[0].T * d_b
    d_weights.insert(0, d_w)

    for layer in range(len(nodes) - 1):
        weights[layer] -= lr * d_weights[layer]
        biases[layer] -= lr * d_biases[layer]

    return weights, biases
