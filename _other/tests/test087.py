import numpy as np


def forward(inputs):
    nodes = [inputs]
    for layer in range(len(layer_sizes) - 1):
        node_layer = activation['f'](nodes[-1] @ weights[layer] + biases[layer])
        nodes.append(node_layer)
    return nodes


def backward(nodes, y):
    # instantiate gradients
    grad_w = _w_zero_grad
    grad_b = _b_zero_grad

    # calculate gradients
    grad_a = g['d'](y, nodes[-1])
    for lyr in range(-1, -len(nodes) + 1, -1):
        grad_b[lyr] = g['d'](nodes[lyr - 1] @ thetas_w[lyr] + thetas_b[lyr]) * grad_a
        grad_w[lyr] = nodes[lyr - 1].T * grad_b[lyr]
        grad_a = np.sum(thetas_w[lyr] * grad_b[lyr], axis=1)
    grad_b[0] = g['d'](nodes[0] @ thetas_w[0] + thetas_b[0]) * grad_a
    grad_w[0] = nodes[0].T * grad_b[0]

    # return gradients
    return grad_w, grad_b
