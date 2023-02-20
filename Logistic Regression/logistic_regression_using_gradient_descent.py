import numpy as np

def sigmoid(z):
    """
    Calculates the sigmoid function of the input z.

    Parameters:
    - z (float): A scalar or NumPy array of any shape.

    Returns:
    - (float): The sigmoid function of z.
    """
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, alpha=0.01, lambda_=0.1, num_iterations=1000):
    """
    Fits a logistic regression model to the input data using gradient descent with L2 regularization.

    Parameters:
    - X (array-like): An array of shape (n_samples, n_features) containing the input features.
    - y (array-like): An array of shape (n_samples,) containing the binary target values (0 or 1).
    - alpha (float): Learning rate for the gradient descent algorithm.
    - lambda_ (float): Regularization parameter for L2 regularization.
    - num_iterations (int): The number of iterations to run the gradient descent algorithm.

    Returns:
    - (tuple): A tuple containing the optimized weights and bias parameters of the logistic regression model.
    """
    # Initialize weights and bias to zeros
    m, n = X.shape
    weights = np.zeros((n, 1))
    bias = 0

    # Run gradient descent for num_iterations
    for i in range(num_iterations):
        # Calculate the predicted target values
        z = np.dot(X, weights) + bias
        h = sigmoid(z)

        # Calculate the gradient of the loss function with respect to weights and bias
        dw = (1 / m) * np.dot(X.T, (h - y))
        db = (1 / m) * np.sum(h - y)

        # Add L2 regularization
        dw += (lambda_ / m) * weights

        # Update the weights and bias
        weights -= alpha * dw
        bias -= alpha * db

    # Return the optimized weights and bias
    return weights, bias
