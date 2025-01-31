import networkx as nx

import nodes
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import math

"""
This program uses the nodes structure to practice basic backpropagation.
Made from scratch (No tutorials, no pytorch).
(NetworkX is NOT used for node structure, only to show the neural net as a visual element)
Version: 0.2
Author: Isaac Park Verbrugge, Christian Host-Madsen
"""

G = nx.Graph()
learning_rate = 0.001


def softmax(outputnodesactivations):
    denom = 0
    eoutputnodesactivations = np.exp(outputnodesactivations)

    for i in eoutputnodesactivations:
        denom += i[0]

    smaxoutputs = np.divide(eoutputnodesactivations, denom)
    return smaxoutputs


def ssr(outputs, correctoutputs):
    correctoutputs = np.array(correctoutputs)
    correctoutputs = correctoutputs.reshape((len(correctoutputs), 1))
    result = np.subtract(correctoutputs, outputs)
    result = np.power(result, 2)
    result = np.sum(result)
    return result


def derivative_ssr(outputs, correctoutputs):
    correctoutputs = np.array(correctoutputs)
    correctoutputs = correctoutputs.reshape((len(correctoutputs), 1))
    result = np.subtract(correctoutputs, outputs)
    result = np.multiply(result, -2)
    return result


def relu(x):  # Leaky Relu function, with a slope of 0.1 for neg values built for use with matrices
    return np.maximum(0.1 * x, x)


def derivative_relu(scalar_passed_to_relu):  # derivative of our leaky relu
    return 1 if (scalar_passed_to_relu > 0) else 0.1


def create_graph(graph, number_input_nodes, number_hidden_layers, number_nodes_per_layer, number_output_nodes):  # graph creation
    name = 0

    # layer matrix sizing
    for i in range(number_hidden_layers):
        graph.layers.append([])
        graph.layers_activations.append([])
        graph.layers_sums.append([])

    # input node creation
    for i in range(number_input_nodes):
        graph.layers[0].append(nodes.Node(name, graph, 0, energy=0, bias=0))
        graph.layers_activations[0].append(graph.layers[0][i].activationEnergy)

        G.add_node(name, pos=(0, i))
        name += 1

    # hidden node creation
    for l in range(number_hidden_layers):
        current_layer = l + 1
        for n in range(number_nodes_per_layer):
            graph.layers[current_layer].append(nodes.Node(name, graph, current_layer, energy=0, bias=0))
            graph.layers_activations[current_layer].append(graph.layers[current_layer][n].activationEnergy)

            G.add_node(name, pos=(current_layer, n))
            name += 1

    # output node creation
    for o in range(number_output_nodes):
        last_layer = len(graph.layers) - 1
        graph.layers[last_layer].append(nodes.Node(name, graph, last_layer, energy=0, bias=0))
        graph.layers_activations[last_layer].append(graph.layers[last_layer][o].activationEnergy)

        G.add_node(name, pos=(last_layer, o))
        name += 1

    # connections
    for i in range(len(graph.layers) - 1):
        for originnode in graph.layers[i]:
            for destinationnode in graph.layers[i + 1]:
                originnode.new_connection(originnode, destinationnode,
                                          np.random.normal(0, 1) * math.sqrt(2/len(graph.layers[i])))  # Xavier Initialization

                G.add_edge(int(originnode.name),int(destinationnode.name))
            originnode.fix_connections_weights()

    # np stuff for np.matmult()
    for i in range(len(graph.layers)):
        graph.layers[i] = np.array(graph.layers[i])
        graph.layers_activations[i] = np.array(graph.layers_activations[i], dtype=object)
        graph.layers_sums[i] = np.array(graph.layers_sums[i], dtype=object)


def forward(graph, inputs):  # forward pass

    def inputstep(inputs):
        # flips inputs so instead of being 1 x n it will be n x 1
        inputs = np.array(inputs).reshape((len(inputs), 1))

        # sets activationEnergies of each object in inputs if we need them for backprop
        for node in range(len(graph.layers[0])):
            graph.layers[0][node].activationEnergy = inputs[node][0]

        # sets the matrix in graphs representing inputs to the n x 1 inputs we found earlier
        graph.layers_activations[0] = inputs

        # makes a np matrix of all the weights we're going to need
        layerweights_initial = []
        for node in graph.layers[0]:
            layerweights_initial += [node.connections_weights]
        layerweights_initial = flipMatrix(layerweights_initial)

        # makes a np matrix of all the biases we're going to need
        bias_initialplusone = []
        for node in graph.layers[1]:
            bias_initialplusone.append([node.bias])
        bias_initialplusone = np.array(bias_initialplusone)

        # weight step
        result = np.matmul(layerweights_initial, graph.layers_activations[0])
        graph.layers_sums[1] = result

        # bias step
        result = result + bias_initialplusone

        # Relu step
        result = relu(result)

        return result

    def layerstep(result):
        for l in range(len(graph.layers) - 2):  # -2: 1 disregarded bc of input, 1 disregarded bc output
            current_layer = l + 1

            # sets activationEnergies of each object in each layer if we need them for backprop
            for node in range(len(graph.layers[current_layer])):
                graph.layers[current_layer][node].activationEnergy = result[node][0]

            # sets the matrix in graphs representing energies of the current layer to the n x 1 results of the last calc
            graph.layers_activations[current_layer] = result

            # sets nodes energies to result
            for node in range(len(graph.layers[current_layer])):
                graph.layers[current_layer][node].energy = result[node][0]

            # makes a np matrix of all the weights we're going to need
            layerweights = []
            for node in graph.layers[current_layer]:  # nodes starting after [0] (inputs)
                layerweights += [node.connections_weights]
            layerweights = flipMatrix(layerweights)

            # makes a np matrix of all the biases we're going to need
            bias_plusone = []
            for node in graph.layers[current_layer + 1]:
                bias_plusone.append([node.bias])
            bias_plusone = np.array(bias_plusone)

            # weight step
            result = np.matmul(layerweights, graph.layers_activations[current_layer])
            graph.layers_sums[current_layer + 1] = result

            # bias step
            result = result + bias_plusone

            # relu step
            result = relu(result)

        return result

    def last_step(result):
        last_layer = len(graph.layers) - 1
        # sets activationEnergies of each object in each layer if we need them for backprop

        for node in range(len(graph.layers[last_layer])):
            graph.layers[last_layer][node].activationEnergy = result[node][0]

        # sets the matrix in graphs representing energies of the current layer to the n x 1 results of the last calcs
        graph.layers_activations[last_layer] = result

        # sets nodes energies to the result
        for node in range(len(graph.layers[last_layer])):
            graph.layers[last_layer][node].energy = result[node][0]

        return result

    # this is kinda crazy. Put 'em all into functional, so it'd be nicer to read. Hope it still works!
    return last_step(layerstep(inputstep(inputs)))


def weightUpdate(connection_in_question, new_weight):
    connection_in_question.weight = new_weight
    connection_in_question.origin.fix_connections_weights()


def biasUpdate(node_in_question, new_bias):
    node_in_question.bias = new_bias


def flipMatrix(in_matrix):  # Function for flipping dimensions of a matrix
    out_matrix = []

    for i in in_matrix[0]:
        out_matrix.append([i])
    for i in range(len(in_matrix) - 1):
        for x in range(len(in_matrix[i + 1])):
            out_matrix[x].append(in_matrix[i + 1][x])
    return np.array(out_matrix)


def newValue(old_value, gradient):
    return old_value - (learning_rate * gradient)


def backward(graph, output_vector, correct_vector):
    g_upstream = derivative_ssr(output_vector, correct_vector)
    # print(g_upstream)

    def last_back():
        nonlocal g_upstream
        g_upstream = np.minimum(1, g_upstream)  # gradient clipping
        last_layer = len(graph.layers) - 1

        g_local = []
        for nodeindex in range(len(graph.layers[last_layer])):  # output bias backprop
            node = graph.layers[last_layer][nodeindex]

            g_relu = derivative_relu(graph.layers_sums[last_layer][nodeindex] + node.bias) * g_upstream[nodeindex][0]
            g_bias = 1 * g_relu

            biasUpdate(node, newValue(node.bias, g_bias))

            g_local.append([g_bias])

        g_upstream = np.array(g_local)

    def layer_back():
        nonlocal g_upstream
        g_upstream = np.minimum(1, g_upstream)  # gradient clipping

        for layer in range(len(graph.layers) - 2, 0, -1):
            g_local = []
            for nodeindex in range(len(graph.layers[layer])):  # weight backprop
                node = graph.layers[layer][nodeindex]

                node_g_activation_path_sum = 0
                for connectionindex in range(len(node.connections)):
                    connection = node.connections[connectionindex]
                    # print(connection)

                    g_weight = node.activationEnergy * g_upstream[connectionindex][0]

                    g_activation = connection.weight * g_upstream[connectionindex][0]

                    weightUpdate(connection, newValue(connection.weight, g_weight))

                    node_g_activation_path_sum += g_activation

                g_local.append([node_g_activation_path_sum])
            # print(g_local)

            g_upstream = np.array(g_local)

            g_local = []
            for nodeindex in range(len(graph.layers[layer])):
                node = graph.layers[layer][nodeindex]

                g_relu = derivative_relu(graph.layers_sums[layer][nodeindex] + node.bias) * g_upstream[nodeindex][0]
                g_bias = 1 * g_relu

                biasUpdate(node, newValue(node.bias, g_bias))

                g_local.append([g_bias])

            g_upstream = np.array(g_local)

    def input_weights_back():
        nonlocal g_upstream
        g_upstream = np.minimum(1, g_upstream)  # gradient clipping

        for nodeindex in range(len(graph.layers[0])):  # weight backprop
            node = graph.layers[0][nodeindex]

            for connectionindex in range(len(node.connections)):
                connection = node.connections[connectionindex]

                g_weight = node.activationEnergy * g_upstream[connectionindex][0]

                # if connectionindex == 0 and nodeindex == 0:
                #     print(g_weight)

                weightUpdate(connection, newValue(connection.weight, g_weight))

    last_back()
    layer_back()
    input_weights_back()


mygraph = nodes.Graph("mygraph")


def testgraphset():
    # Node 1
    weightUpdate(mygraph.layers[0][0].connections[0], 1)
    weightUpdate(mygraph.layers[0][0].connections[1], 1)
    # Node 2
    weightUpdate(mygraph.layers[0][1].connections[0], 1)
    weightUpdate(mygraph.layers[0][1].connections[1], 1)

    # Node 3
    biasUpdate(mygraph.layers[1][0], 0)
    weightUpdate(mygraph.layers[1][0].connections[0], 1)
    weightUpdate(mygraph.layers[1][0].connections[1], 1)
    # Node 4
    biasUpdate(mygraph.layers[1][1], 0)
    weightUpdate(mygraph.layers[1][1].connections[0], 1)
    weightUpdate(mygraph.layers[1][1].connections[1], 1)

    # Node 6
    biasUpdate(mygraph.layers[2][0], 0)
    # Node 7
    biasUpdate(mygraph.layers[2][1], 0)


create_graph(mygraph, 2, 3, 4, 3)  # graph, number_input_nodes, number_hidden_layers, number_nodes_per_layer, number_output_nodes

# testgraphset()

data = [
    [[1, 0], [1, 0, 0]], [[1, 1], [0.2, 1, 0.2]], [[0, 1], [0, 0, 1]], [[0.1, 1.2], [0, 0.1, 1]], [[0.2, 0.8], [0.1, 0.3, 1]], [[0.5, 0.5], [0.5, 1, 0.5]], [[0.45, 0.65], [0.22, 1, 0.33]], [[0.7, 0], [0.7, 0, 0]], [[0.6, 0.1], [0.6, 0, 0.1]], [[10, 1], [1, 0, 0.01]]
]

# weightx = np.linspace(-2,2)
# error2y = []
# for i in weightx:
#     weightUpdate(mygraph.layers[0][0].connections[0], i)
#     error = 0
#     for point in processed_data:
#         calculatedoutputs = forward(mygraph, point[0])
#         error += ssr(calculatedoutputs, point[1])
#     error /= len(processed_data)
#     error2y.append(error)
# plt.plot(weightx, error2y)
# plt.show()
# weightUpdate(mygraph.layers[0][0].connections[0], 0)

def training(epoch):
    for point in data:
        calculatedoutputs = forward(mygraph, point[0])
        backward(mygraph, calculatedoutputs, point[1])

    error = 0
    for point in data:
        calculatedoutputs = forward(mygraph, point[0])
        error += ssr(calculatedoutputs, point[1])
    error /= len(data)

    # for mylayer in mygraph.layers:
    #     for mynode in mylayer:
    #         print(mynode.bias, end=" ")
    #         for myconnection in mynode.connections_weights:
    #             print(myconnection, sep="", end="~")
    #         print(end="\n\n")

    epochx.append(epoch)
    errory.append(error)

    plt.cla()
    plt.plot(epochx, errory)


epochs = 500
errory = []
epochx = []
ani = FuncAnimation(plt.gcf(), training, interval=0, frames=epochs, repeat=0)

plt.show()

# MSE step
error = 0
for point in data:
    calculatedoutputs = forward(mygraph, point[0])
    error += ssr(calculatedoutputs, point[1])
error /= len(data)
print(f"MSE: {error}")

# Unknown value test
while True:
    user_test = []
    for i in range(len(mygraph.layers[0])):
        activation = float(input(f"activation for node {i}: "))
        user_test.append([activation])
    print(forward(mygraph, user_test))


pos=nx.get_node_attributes(G,'pos')
nx.draw(G, pos, with_labels=True)
plt.show()






