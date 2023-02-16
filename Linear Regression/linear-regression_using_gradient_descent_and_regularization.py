import numpy as np

def fit_linear_regression(X, y, alpha, lambda_, num_iterations):
    """
    Fits a linear regression model to the input data using gradient descent with L2 regularization.

    Parameters:
    X (array-like): An array of shape (n_samples, n_features) containing the input features.
    y (array-like): An array of shape (n_samples,) containing the target values.
    alpha (float): Learning rate for the gradient descent algorithm.
    lambda_ (float): Regularization parameter for L2 regularization.
    num_iterations (int): The number of iterations to run the gradient descent algorithm.

    Returns:
    tuple: A tuple containing the slope and intercept coefficients of the linear regression model.
    """
    # Scale the input features
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # Get the number of samples and features
    n, m = X.shape
    # Initialize the theta coefficients to 0
    theta = np.zeros(m+1)
    # Add a column of ones to the input features for the intercept term
    X = np.column_stack((np.ones(n), X_scaled))

    # Run gradient descent for the given number of iterations
    for i in range(num_iterations):
        # Compute the predicted y values
        h = np.dot(X, theta)
        # Compute the loss (difference between predicted y and true y)
        loss = h - y
        # Compute the gradient of the cost function with L2 regularization
        gradient = np.dot(X.T, loss) / n + (lambda_ / n) * np.append(0, theta[1:])
        # Update the theta coefficients using the gradient descent algorithm
        theta = theta - alpha * gradient

    # Return the slope and intercept coefficients
    return theta[1:], theta[0]

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