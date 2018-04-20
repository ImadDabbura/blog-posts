"""
Implementing three different parameters initialization methods.
"""

import matplotlib.pyplot as plt
import numpy as np

from .coding_neural_network_from_scratch import (L_model_forward,
                                                 compute_cost,
                                                 L_model_backward,
                                                 update_parameters,
                                                 accuracy)


def initialize_parameters_zeros(layers_dims):
    """
    Initializes the parameters dictionary to all zeros for both weights and
    bias.

    Arguments
    ---------
    layer_dims : list
        input size and size of each layer, length: number of layers + 1.

    Returns
    -------
    parameters : dict
        weight matrix and the bias vector for each layer.
    """
    np.random.seed(1)               
    parameters = {}                 
    L = len(layers_dims)            

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros(
            (layers_dims[l], layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_random(layers_dims):
    """
    Initializes the parameters dictionary rabdomly from standard normal
    distribution multiplied by 10 for weight matrices and zeros for bias
    vectors.

    Arguments
    ---------
    layer_dims : list
        input size and size of each layer, length: number of layers + 1.

    Returns
    -------
    parameters : dict
        weight matrix and the bias vector for each layer.
    """
    np.random.seed(1)               
    parameters = {}                 
    L = len(layers_dims)            

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l - 1]) * 10
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_he_xavier(layers_dims, initialization_method="he"):
    """
    Initializes the parameters dictionary for weights based on "He" and
    "Xavier" methods and zeros for bias vectors.

    Arguments
    ---------
    layer_dims : list
        input size and size of each layer, length: number of layers + 1.
    initialization_method : str
        specify the initialization method to be used: "he", "xavier".

    Returns
    -------
    parameters : dict
        weight matrix and the bias vector for each layer.
    """
    np.random.seed(1)               
    parameters = {}                 
    L = len(layers_dims)            

    if initialization_method == "he":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(
                layers_dims[l],
                layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    elif initialization_method == "xavier":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(
                layers_dims[l],
                layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def model(X, Y, layers_dims, learning_rate=0.01, num_iterations=1000,
          print_cost=True, hidden_layers_activation_fn="relu",
          initialization_method="he"):
    """
    Implements multilayer neural network using gradient descent as the
    learning algorithm.

    Arguments
    ---------
    X : 2d-array
        data, shape: number of examples x num_px * num_px * 3.
    y : 2d-array
        true "label" vector, shape: 1 x number of examples.
    layers_dims : list
        input size and size of each layer, length: number of layers + 1.
    learning_rate : float
        learning rate of the gradient descent update rule.
    num_iterations : int
        number of iterations of the optimization loop.
    print_cost : bool
        if True, it prints the cost every 100 steps.
    hidden_layers_activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".
    initialization_method : str
        specify the initialization method to be used: "he", "xavier".

    Returns
    -------
    parameters : dict
        parameters learnt by the model. They can then be used to predict test
        examples.
    """
    np.random.seed(1)

    # initialize cost list
    cost_list = []

    # initialize parameters
    if initialization_method == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization_method == "random":
        parameters = initialize_parameters_random(layers_dims)
    else:
        parameters = initialize_parameters_he_xavier(
            layers_dims, initialization_method)

    # iterate over num_iterations
    for i in range(num_iterations):
        # iterate over L-layers to get the final output and the cache
        AL, caches = L_model_forward(
            X, parameters, hidden_layers_activation_fn)

        # compute cost to plot it
        cost = compute_cost(AL, Y)

        # iterate over L-layers backward to get gradients
        grads = L_model_backward(AL, Y, caches, hidden_layers_activation_fn)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # append each 100th cost to the cost list
        if (i + 1) % 100 == 0 and print_cost:
            print("The cost after {} iterations is: {}".format(i + 1, cost))

        if i % 100 == 0:
            cost_list.append(cost)

    # plot the cost curve
    plt.plot(cost_list)
    plt.xlabel("Iterations (per hundreds)")
    plt.ylabel("Cost")
    plt.title(
        "Cost curve: learning rate = {} and {} initialization method".format(
            learning_rate, initialization_method))

    return parameters
