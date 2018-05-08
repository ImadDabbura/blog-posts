"""
Implement L2 regularization of a fully connected neural network.
"""


import matplotlib.pyplot as plt
import numpy as np

from coding_neural_network_from_scratch import (initialize_parameters,
                                                L_model_forward,
                                                compute_cost,
                                                relu_gradient,
                                                sigmoid_gradient,
                                                tanh_gradient,
                                                update_parameters,
                                                accuracy)
from gradient_checking import dictionary_to_vector


def compute_cost_reg(AL, y, parameters, lambd=0):
    """
    Computes the Cross-Entropy cost function with L2 regularization.

    Arguments
    ---------
    AL : 2d-array
        probability vector of shape 1 x training_examples.
    y : 2d-array
        true "label" vector.
    parameters : dict
        contains all the weight matrices and bias vectors for all layers.
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    cost : float
        binary cross-entropy cost.
    """
    # number of examples
    m = y.shape[1]

    # compute traditional cross entropy cost
    cross_entropy_cost = compute_cost(AL, y)

    # convert parameters dictionary to vector
    parameters_vector = dictionary_to_vector(parameters)

    # compute the regularization penalty
    L2_regularization_penalty = (
        lambd / (2 * m)) * np.sum(np.square(parameters_vector))

    # compute the total cost
    cost = cross_entropy_cost + L2_regularization_penalty

    return cost


def linear_backword_reg(dZ, cache, lambd=0):
    """
    Computes the gradient of the output w.r.t weight, bias, & post-activation
    output of (l - 1) layers at layer l.

    Arguments
    ---------
    dZ : 2d-array
        gradient of the cost w.r.t. the linear output (of current layer l).
    cache : tuple
        values of (A_prev, W, b) coming from the forward propagation in the
        current layer.
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    dA_prev : 2d-array
        gradient of the cost w.r.t. the activation (of the previous layer l-1).
    dW : 2d-array
        gradient of the cost w.r.t. W (current layer l).
    db : 2d-array
        gradient of the cost w.r.t. b (current layer l).
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambd / m) * W
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward_reg(dA, cache, activation_fn="relu", lambd=0):
    """
    Arguments
    ---------
    dA : 2d-array
        post-activation gradient for current layer l.
    cache : tuple
        values of (linear_cache, activation_cache).
    activation : str
        activation used in this layer: "sigmoid", "tanh", or "relu".
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    dA_prev : 2d-array
        gradient of the cost w.r.t. the activation (of previous layer l-1),
        same shape as A_prev.
    dW : 2d-array
        gradient of the cost w.r.t. W (current layer l), same shape as W.
    db : 2d-array
        gradient of the cost w.r.t. b (current layer l), same shape as b.
    """
    linear_cache, activation_cache = cache

    if activation_fn == "sigmoid":
        dZ = sigmoid_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword_reg(dZ, linear_cache, lambd)

    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword_reg(dZ, linear_cache, lambd)

    elif activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword_reg(dZ, linear_cache, lambd)

    return dA_prev, dW, db


def L_model_backward_reg(AL, y, caches, hidden_layers_activation_fn="relu",
                         lambd=0):
    """
    Computes the gradient of output layer w.r.t weights, biases, etc. starting
    on the output layer in reverse topological order.

    Arguments
    ---------
    AL : 2d-array
        probability vector, output of the forward propagation
        (L_model_forward()).
    y : 2d-array
        true "label" vector (containing 0 if non-cat, 1 if cat).
    caches : list
        list of caches for all layers.
    hidden_layers_activation_fn :
        activation function used on hidden layers: "tanh", "relu".
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    grads : dict
        gradients.
    """
    y = y.reshape(AL.shape)
    L = len(caches)
    grads = {}

    dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] =\
        linear_activation_backward_reg(dAL, caches[L - 1], "sigmoid", lambd)

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] =\
            linear_activation_backward_reg(
                grads["dA" + str(l)], current_cache,
                hidden_layers_activation_fn, lambd)

    return grads


def model_with_regularization(
        X, y, layers_dims, learning_rate=0.01,  num_epochs=3000,
        print_cost=False, hidden_layers_activation_fn="relu", lambd=0):
    """
    Implements L-Layer neural network.

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
     num_epochs : int
        number of times to over the training data.
    print_cost : bool
        if True, it prints the cost every 100 steps.
    hidden_layers_activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".
    lambd : float
        regularization hyperparameter.

    Returns
    -------
    parameters : dict
        parameters learnt by the model. They can then be used to predict test
        examples.
    """
    # get number of examples
    m = X.shape[1]

    # to get consistents output
    np.random.seed(1)

    # initialize parameters
    parameters = initialize_parameters(layers_dims)

    # intialize cost list
    cost_list = []

    # implement gradient descent
    for i in range(num_epochs):
        # compute forward propagation
        AL, caches = L_model_forward(
            X, parameters, hidden_layers_activation_fn)

        # compute regularized cost
        reg_cost = compute_cost_reg(AL, y, parameters, lambd)

        # compute gradients
        grads = L_model_backward_reg(
            AL, y, caches, hidden_layers_activation_fn, lambd)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # print cost
        if (i + 1) % 100 == 0 and print_cost:
            print("The cost after {} iterations: {}".format(
                (i + 1), reg_cost))

        # append cost
        if i % 100 == 0:
            cost_list.append(reg_cost)

    # plot the cost curve
    plt.plot(cost_list)
    plt.xlabel("Iterations (per hundreds)")
    plt.ylabel("Cost")
    plt.title("Cost curve for the learning rate = {}".format(learning_rate))

    return parameters
