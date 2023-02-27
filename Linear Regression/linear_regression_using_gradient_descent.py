import numpy as np

def fit_linear_regression(X, y, alpha, num_iterations):
    """
    Fits a linear regression model to the input data using gradient descent.

    Parameters:
    X (array-like): An array of shape (n_samples,) containing the input features.
    y (array-like): An array of shape (n_samples,) containing the target values.
    alpha (float): Learning rate for the gradient descent algorithm.
    num_iterations (int): The number of iterations to run the gradient descent algorithm.

    Returns:
    tuple: A tuple containing the slope and intercept coefficients of the linear regression model.
    """
    # Get the number of samples
    n = len(X)
    # Initialize the theta coefficients to 0
    theta = np.zeros(2)
    # Add a column of ones to the input features for the intercept term
    X = np.vstack((np.ones(n), X)).T

    # Run gradient descent for the given number of iterations
    for i in range(num_iterations):
        # Compute the predicted y values
        h = np.dot(X, theta)
        # Compute the loss (difference between predicted y and true y)
        loss = h - y
        # Compute the gradient of the cost function
        gradient = np.dot(X.T, loss) / n
        # Update the theta coefficients using the gradient descent algorithm
        theta = theta - alpha * gradient

    # Return the slope and intercept coefficients
    return theta[1], theta[0]

def predict_linear_regression(X, slope, intercept):
    """
    Predicts target values for new input data using the slope and intercept coefficients.

    Parameters:
    X (array-like): An array of shape (n_samples,) containing the input features.
    slope (float): The slope coefficient of the linear regression model.
    intercept (float): The intercept coefficient of the linear regression model.

    Returns:
    array: An array of shape (n_samples,) containing the predicted target values.
    """
    # Compute the predicted target values
    y_pred = slope * X + intercept
    # Return the predicted target values
    return y_pred

