"""
Implementation of Character-Level Language Model on names.
"""
# Load packages
import numpy as np


def initialize_parameters(vocab_size, hidden_layer_size):
    """
    Initialze model's parameters. biases will be initialzed to zeros and
    weights will be initialized to small random numbers from standard normal
    distribution.

    Arguments
    ---------
    vocab_size : int
        size of the vocabulary dictionary.
    hidden_layer_size : int
        size of hidden units.

    Returns
    -------
    parameters : python dict
        dictionary containing all the initialized parameters.
            Whh -- hidden to hidden.
            Wxh -- input to hidden.
            b -- hidden bias.
            Why -- hidden to output.
            c -- output bias.
    """
    parameters = {}
    parameters["Whh"] = np.random.randn(
        hidden_layer_size, hidden_layer_size) * 0.01
    parameters["Wxh"] = np.random.randn(hidden_layer_size, vocab_size) * 0.01
    parameters["b"] = np.zeros((hidden_layer_size, 1))
    parameters["Why"] = np.random.randn(vocab_size, hidden_layer_size) * 0.01
    parameters["c"] = np.zeros((vocab_size, 1))

    return parameters


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "Whh", "Wxh", "b", "Why", "c".
                - values: numpy arrays of zeros of the same shape as the
                          corresponding gradients/parameters.

    Arguments
    ---------
    parameters : python dict
        dictionary containing all the parameters.

    Returns
    -------
    v : python dict
        dictionary that will contain the exponentially weighted average of the
        gradients.
    s : python dict
        dictionary that will contain the exponentially weighted average of the
        squared gradients.
    """
    parameters_names = ["Whh", "Wxh", "b", "Why", "c"]
    v = {}
    s = {}

    for param_name in parameters_names:
        v["d" + param_name] = np.zeros_like(parameters[param_name])
        s["d" + param_name] = np.zeros_like(parameters[param_name])

    return v, s


def initialize_rmsprop(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "Whh", "Wxh", "b", "Why", "c".
                - values: numpy arrays of zeros of the same shape as the
                          corresponding gradients/parameters.

    Arguments
    ---------
    parameters : python dict
        dictionary containing all parameters.

    Returns
    -------
    s : python dict
        dictionary the exponentially weighted average of squared gradients.
    """
    parameters_names = ["Whh", "Wxh", "b", "Why", "c"]
    s = {}

    for param_name in parameters_names:
        s["d" + param_name] = np.zeros_like(parameters[param_name])

    return s


def softmax(z):
    """
    Implements softmax on the array z and returns normalized probability.

    Arguments
    ---------
    z : array-like
        array contains logits.

    Returns
    -------
    probs : array
        array containg the probability of each element from the logits array.

    """
    e_z = np.exp(z)
    probs = e_z / np.sum(e_z)

    return probs


def rnn_forward(x, y, h_prev, parameters):
    """
    Implement one Forward pass on one name.

    Arguments
    ---------
    x : list
        list of integers for the index of the characters in the example
        shifted one character to the right.
    y : list
        list of integers for the index of the characters in the example.
    h_prev : array
        last hidden state from the previous example.
    parameters : python dict
        dictionary containing the parameters.

    Returns
    -------
    loss : float
        cross-entropy loss.
    cache : tuple
        contains three python dictionaries:
            xs -- input of all time steps.
            hs -- hidden state of all time steps.
            probs -- probability distribution of each character at each time
                step.
    """
    # Retrieve parameters
    Wxh, Whh, b = parameters["Wxh"], parameters["Whh"], parameters["b"]
    Why, c = parameters["Why"], parameters["c"]

    # Initialize inputs, hidden state, output, and probabilities dictionaries
    xs, hs, os, probs = {}, {}, {}, {}

    # Initialize x0 to zero vector
    xs[0] = np.zeros((vocab_size, 1))

    # Initialize loss and assigns h_prev to last hidden state in hs
    loss = 0
    hs[-1] = np.copy(h_prev)

    # Forward pass: loop over all characters of the name
    for t in range(len(x)):
        # Convert to one-hot vector
        if t > 0:
            xs[t] = np.zeros((vocab_size, 1))
            xs[t][x[t]] = 1
        # Hidden state
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + b)
        # Logits
        os[t] = np.dot(Why, hs[t]) + c
        # Probs
        probs[t] = softmax(os[t])
        # Loss
        loss -= np.log(probs[t][y[t], 0])

    cache = (xs, hs, probs)

    return loss, cache


def smooth_loss(loss, current_loss):
    """
    Compute the weighted average of the loss to smooth it out.
    """
    return 0.999 * loss + 0.001 * current_loss


def clip_gradients(gradients, max_value):
    """
    Implements gradient clipping element-wise on gradients to be between the
    interval [-max_value, max_value].

    Arguments
    ----------
    gradients : python dict
        dictionary that stores all the gradients.
    max_value : scalar
        edge of the interval [-max_value, max_value].

    Returns
    -------
    gradients : python dict
        dictionary where all gradients were clipped.
    """
    for grad in gradients.keys():
        np.clip(gradients[grad], -max_value, max_value, out=gradients[grad])

    return gradients


def rnn_backward(y, parameters, cache):
    """
    Implements Backpropagation on one name.

    Arguments
    ---------
    y : list
        list of integers for the index of the characters in the example.
    parameters : python dict
        dictionary containing the parameters.
    cache : tuple
            contains three python dictionaries:
                xs -- input of all time steps.
                hs -- hidden state of all time steps.
                probs -- probability distribution of each character at each time
                    step.

    Returns
    -------
    grads : python dict
        dictionary containing all the gradients.
    h_prev : array
        last hidden state from the current example.
    """
    # Retrieve xs, hs, and probs
    xs, hs, probs = cache

    # Initialize all gradients to zero
    dh_next = np.zeros_like(hs[0])

    parameters_names = ["Whh", "Wxh", "b", "Why", "c"]
    grads = {}
    for param_name in parameters_names:
        grads["d" + param_name] = np.zeros_like(parameters[param_name])

    # Iterate over all time steps in reverse order starting from Tx
    for t in reversed(range(len(xs))):
        dy = np.copy(probs[t])
        dy[y[t]] -= 1
        grads["dWhy"] += np.dot(dy, hs[t].T)
        grads["dc"] += dy
        dh = np.dot(parameters["Why"].T, dy) + dh_next
        dhraw = (1 - hs[t] ** 2) * dh
        grads["dWhh"] += np.dot(dhraw, hs[t - 1].T)
        grads["dWxh"] += np.dot(dhraw, xs[t].T)
        grads["db"] += dhraw
        dh_next = np.dot(parameters["Whh"].T, dhraw)
        # Clip the gradients using [-5, 5] as the interval
        grads = clip_gradients(grads, 5)
    # Get the last hidden state
    h_prev = hs[len(xs) - 1]

    return grads, h_prev


def update_parameters_with_adam(
        parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999,
        epsilon=1e-8):
    """
    Update parameters using Adam.

    Arguments
    ---------
    parameters : python dict
        dictionary containing all parameters.
    grads : python dict
        dictionary containing gradients for all parameters.
    v : python dict
        Adam variable, moving average of the first gradient.
    s : python dict
        Adam variable, moving average of the squared gradient.
    learning_rate : float
        learning rate step size.
    beta1 : float
        exponential decay hyperparameter for the first moment estimates.
    beta2 : float
        exponential decay hyperparameter for the second moment estimates.
    epsilon : float
        hyperparameter preventing division by zero in Adam updates.

    Returns
    -------
    parameters : python dict
        dictionary containing updated parameters.
    v : python dict
        Adam variable, moving average of the first gradient.
    s : python dict
        Adam variable, moving average of the squared gradient.
    """
    parameters_names = ["Whh", "Wxh", "b", "Why", "c"]
    v_corrected = {}
    s_corrected = {}

    for param_name in parameters_names:
        # Update the moving average of first gradient and squared gradient
        v["d" + param_name] = beta1 * v["d" + param_name] +\
            (1 - beta1) * grads["d" + param_name]
        s["d" + param_name] = beta2 * s["d" + param_name] +\
            (1 - beta2) * np.square(grads["d" + param_name])

        # Compute the corrected-bias estimate of the moving averages
        v_corrected["d" + param_name] = v["d" + param_name] / (1 - beta1**t)
        s_corrected["d" + param_name] = s["d" + param_name] / (1 - beta2**t)

        # update parameters
        parameters[param_name] -= (learning_rate *
                                   v_corrected["d" + param_name])\
            / (np.sqrt(s_corrected["d" + param_name] + epsilon))

    return parameters, v, s


def update_parameters(parameters, grads, learning_rate):
    for param in parameters.keys():
        parameters[param] -= learning_rate * grads["d" + param]

    return parameters


def update_parameters_with_rmsprop(
        parameters, grads, s, beta=0.9, learning_rate=0.001, epsilon=1e-8):
    """
    Update parameters using RMSProp.

    Arguments
    ---------
    parameters : python dict
        dictionary containing all parameters.
    grads : python dict
        dictionary containing gradients for all parameters.
    s : python dict
        dictionary containing the exponential weighted average of squared
        gradients.
    learning_rate : float
        learning rate step size.
    beta : float
        the momentum hyperparameter.
    learning_rate : float
        the learning rate.
    epsilon : float
         hyperparameter preventing division by zero in parameter updates.

    Returns
    -------
    parameters : python dict
        python dictionary containing updated parameters.
    s : python dict
        python dictionary containing updated exponential weighted average of
        squared gradients.
    """
    parameters_names = ["Whh", "Wxh", "b", "Why", "c"]

    for param_name in parameters_names:
        # Update exponential weighted average of squared gradients
        s["d" + param_name] = beta * s["d" + param_name] +\
            (1 - beta) * np.square(grads["d" + param_name])

        # Update parameters
        parameters[param_name] -= (learning_rate * grads["d" + param_name])\
            / (np.sqrt(s["d" + param_name] + epsilon))

    return parameters, s


def sample(parameters, idx_to_chars, chars_to_idx, n):
    """
    Implements sampling of a squence of n characters characters length. The
    sampling will be based on the probability distribution output of RNN.

    Arguments
    ---------
    parameters : python dict
        dictionary storing all the parameters of the model.
    idx_to_chars : python dict
        dictionary mapping indices to characters.
    chars_to_idx : python dict
        dictionary mapping characters to indices.
    n : scalar
        number of characters to output.

    Returns
    -------
    sequence : str
        sequence of characters sampled.
    """
    # Retrienve parameters, shapes, and vocab size
    Whh, Wxh, b = parameters["Whh"], parameters["Wxh"], parameters["b"]
    Why, c = parameters["Why"], parameters["c"]
    n_h, n_x = Wxh.shape
    vocab_size = c.shape[0]

    # Initialize a0 and x1 to zero vectors
    h_prev = np.zeros((n_h, 1))
    x = np.zeros((n_x, 1))

    # Initialize empty sequence
    indices = []
    idx = -1
    counter = 0
    while (counter <= n and idx != chars_to_idx["\n"]):
        # Fwd propagation
        h = np.tanh(np.dot(Whh, h_prev) + np.dot(Wxh, x) + b)
        o = np.dot(Why, h) + c
        probs = softmax(o)

        # Sample the index of the character using generated probs distribution
        idx = np.random.choice(vocab_size, p=probs.ravel())

        # Get the character of the sampled index
        char = idx_to_chars[idx]

        # Add the char to the sequence
        indices.append(idx)

        # Update a_prev and x
        h_prev = np.copy(h)
        x = np.zeros((n_x, 1))
        x[idx] = 1

        counter += 1
    sequence = "".join([idx_to_chars[idx] for idx in indices if idx != 0])

    return sequence


def model(
        file_path, chars_to_idx, idx_to_chars, hidden_layer_size, vocab_size,
        num_epochs=10, learning_rate=0.01):
    """
    Implements RNN to generate characters.

    Arguments
    ---------
    file_path : str
        path to the file of the raw data.
    num_epochs : int
        number of passes the optimization algorithm to go over the training
        data.
    learning_rate : float
        step size of learning.
    chars_to_idx : python dict
        dictionary mapping characters to indices.
    idx_to_chars : python dict
        dictionary mapping indices to characters.
    hidden_layer_size : int
        number of hidden units in the hidden layer.
    vocab_size : int
        size of vocabulary dictionary.

    Returns
    -------
    parameters : python dict
        dictionary storing all the parameters of the model.
    overall_loss : list
        list stores smoothed loss per epoch.
    """
    # Get the data
    with open(file_path) as f:
        data = f.readlines()
    examples = [x.lower().strip() for x in data]

    # Initialize parameters
    parameters = initialize_parameters(vocab_size, hidden_layer_size)

    # Initialize Adam parameters
    s = initialize_rmsprop(parameters)

    # Initialize loss
    smoothed_loss = -np.log(1 / vocab_size) * 7

    # Initialize hidden state h0 and overall loss
    h_prev = np.zeros((hidden_layer_size, 1))
    overall_loss = []

    # Iterate over number of epochs
    for epoch in range(num_epochs):
        print(f"\033[1m\033[94mEpoch {epoch}")
        print(f"\033[1m\033[92m=======")

        # Sample one name
        print(f"""Sampled name: {sample(parameters, idx_to_chars, chars_to_idx,
            10).capitalize()}""")
        print(f"Smoothed loss: {smoothed_loss:.4f}\n")

        # Shuffle examples
        np.random.shuffle(examples)

        # Iterate over all examples (SGD)
        for example in examples:
            x = [None] + [chars_to_idx[char] for char in example]
            y = x[1:] + [chars_to_idx["\n"]]
            # Fwd pass
            loss, cache = rnn_forward(x, y, h_prev, parameters)
            # Compute smooth loss
            smoothed_loss = smooth_loss(smoothed_loss, loss)
            # Bwd pass
            grads, h_prev = rnn_backward(y, parameters, cache)
            # Update parameters
            parameters, s = update_parameters_with_rmsprop(
                parameters, grads, s)

        overall_loss.append(smoothed_loss)

    return parameters, overall_loss
