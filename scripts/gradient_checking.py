"""
Implement gradient checking of fully connected neural network.
"""


import numpy as np
from numpy.linalg import norm

from coding_neural_network_from_scratch import L_model_forward, compute_cost


def dictionary_to_vector(params_dict):
    """
    Roll a dictionary into a single vector.

    Arguments
    ---------
    params_dict : dict
        learned parameters.

    Returns
    -------
    params_vector : array
        vector of all parameters concatenated.
    """
    count = 0
    for key in params_dict.keys():
        new_vector = np.reshape(params_dict[key], (-1, 1))
        if count == 0:
            theta_vector = new_vector
        else:
            theta_vector = np.concatenate((theta_vector, new_vector))
        count += 1

    return theta_vector


def vector_to_dictionary(vector, layers_dims):
    """
    Unroll parameters vector to dictionary using layers dimensions.

    Arguments
    ---------
    vector : array
        parameters vector.
    layers_dims : list or array_like
        dimensions of each layer in the network.

    Returns
    -------
    parameters : dict
        dictionary storing all parameters.
    """
    L = len(layers_dims)
    parameters = {}
    k = 0

    for l in range(1, L):
        # Create temp variable to store dimension used on each layer
        w_dim = layers_dims[l] * layers_dims[l - 1]
        b_dim = layers_dims[l]

        # Create temp var to be used in slicing parameters vector
        temp_dim = k + w_dim

        # add parameters to the dictionary
        parameters["W" + str(l)] = vector[
            k:temp_dim].reshape(layers_dims[l], layers_dims[l - 1])
        parameters["b" + str(l)] = vector[
            temp_dim:temp_dim + b_dim].reshape(b_dim, 1)

        k += w_dim + b_dim

    return parameters


def gradients_to_vector(gradients):
    """
    Roll all gradients into a single vector containing only dW and db.

    Arguments
    ---------
    gradients : dict
        storing gradients of weights and biases for all layers: dA, dW, db.

    Returns
    -------
    new_grads : array
        vector of only dW and db gradients.
    """
    # Get the number of indices for the gradients to iterate over
    valid_grads = [key for key in gradients.keys()
                   if not key.startswith("dA")]
    L = len(valid_grads)// 2
    count = 0
    
    # Iterate over all gradients and append them to new_grads list
    for l in range(1, L + 1):
        if count == 0:
            new_grads = gradients["dW" + str(l)].reshape(-1, 1)
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        else:
            new_grads = np.concatenate(
                (new_grads, gradients["dW" + str(l)].reshape(-1, 1)))
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        count += 1
        
    return new_grads


def forward_prop_cost(X, parameters, Y, hidden_layers_activation_fn="tanh"):
    """
    Implements the forward propagation and computes the cost.
    
    Arguments
    ---------
    X : 2d-array
        input data, shape: number of features x number of examples.
    parameters : dict
        parameters to use in forward prop.
    Y : array
        true "label", shape: 1 x number of examples.
    hidden_layers_activation_fn : str
        activation function to be used on hidden layers: "tanh", "relu".

    Returns
    -------
    cost : float
        cross-entropy cost.
    """
    # Compute forward prop
    AL, _ = L_model_forward(X, parameters, hidden_layers_activation_fn)

    # Compute cost
    cost = compute_cost(AL, Y)

    return cost


def gradient_check(
        parameters, gradients, X, Y, layers_dims, epsilon=1e-7,
        hidden_layers_activation_fn="tanh"):
    """
    Checks if back_prop computes correctly the gradient of the cost output by
    forward_prop.
    
    Arguments
    ---------
    parameters : dict
        storing all parameters to use in forward prop.
    gradients : dict
        gradients of weights and biases for all layers: dA, dW, db.
    X : 2d-array
        input data, shape: number of features x number of examples.
    Y : array
        true "label", shape: 1 x number of examples.
    epsilon : 
        tiny shift to the input to compute approximate gradient.
    layers_dims : list or array_like
        dimensions of each layer in the network.
    
    Returns
    -------
    difference : float
        difference between approx gradient and back_prop gradient
    """
    
    # Roll out parameters and gradients dictionaries
    parameters_vector = dictionary_to_vector(parameters)
    gradients_vector = gradients_to_vector(gradients)

    # Create vector of zeros to be used with epsilon
    grads_approx = np.zeros_like(parameters_vector)

    for i in range(len(parameters_vector)):
        # Compute cost of theta + epsilon
        theta_plus = np.copy(parameters_vector)
        theta_plus[i] = theta_plus[i] + epsilon
        j_plus = forward_prop_cost(
            X, vector_to_dictionary(theta_plus, layers_dims), Y,
            hidden_layers_activation_fn)

        # Compute cost of theta - epsilon
        theta_minus = np.copy(parameters_vector)
        theta_minus[i] = theta_minus[i] - epsilon
        j_minus = forward_prop_cost(
            X, vector_to_dictionary(theta_minus, layers_dims), Y,
            hidden_layers_activation_fn)

        # Compute numerical gradients
        grads_approx[i] = (j_plus - j_minus) / (2 * epsilon)

    # Compute the difference of numerical and analytical gradients
    numerator = norm(gradients_vector - grads_approx)
    denominator = norm(grads_approx) + norm(gradients_vector)
    difference = numerator / denominator

    if difference > 10e-7:
        print ("\033[31mThere is a mistake in back-propagation " +\
               "implementation. The difference is: {}".format(difference))
    else:
        print ("\033[32mThere implementation of back-propagation is fine! "+\
               "The difference is: {}".format(difference))

    return difference
