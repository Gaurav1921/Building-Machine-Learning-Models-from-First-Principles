# Gradient Descent Linear Regression Model from Scratch

This repository contains a simple implementation of a linear regression model using the gradient descent algorithm with L2 regularization, built from scratch in Python without the use of external libraries. The model is designed to take in a set of input features and target values, and outputs a set of slope and intercept coefficients that can be used to predict new target values from new input features.

## Installation

To use this model, simply clone the repository to your local machine:

```sh
git clone https://github.com/<username>/<repository-name>.git
```


## Usage

To use the model, import the fit_linear_regression function from the linear_regression.py module, and pass in your input features and target values as NumPy arrays. Here's an example:

```sh
import numpy as np
from linear_regression import fit_linear_regression, predict_linear_regression

# Generate some sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 5, 7])

# Fit the linear regression model
slope, intercept = fit_linear_regression(X, y, alpha=0.01, lambda_=0.1, num_iterations=1000)

# Predict new target values
X_new = np.array([[7, 8], [9, 10]])
y_new = predict_linear_regression(X_new, slope, intercept)
```

## Documentation

The linear_regression.py module contains two functions:

"fit_linear_regression(X, y, alpha, lambda_, num_iterations)": Fits a linear regression model to the input data using gradient descent with L2 regularization.

### Parameters:

+ X (array-like): An array of shape (n_samples, n_features) containing the input features.
+ y (array-like): An array of shape (n_samples,) containing the target values.
+ alpha (float): Learning rate for the gradient descent algorithm.
+ lambda_ (float): Regularization parameter for L2 regularization.
+ num_iterations (int): The number of iterations to run the gradient descent algorithm.

### Returns:

+ tuple: A tuple containing the slope and intercept coefficients of the linear regression model.

"predict_linear_regression(X, slope, intercept)": Predicts target values for new input features using the slope and intercept coefficients of a linear regression model.

## Modifying the Model
You can modify the existing implementation to potentially improve the accuracy of the linear regression model. Some possible modifications include:

+ Feature scaling: You can scale the input features to have a similar range, which can help the gradient descent algorithm converge more quickly. To do this, you can subtract the mean of each feature and divide by the standard deviation.
+ Regularization: You can add a regularization term to the cost function to prevent overfitting. There are two commonly used types of regularization: L1 regularization (also known as Lasso) and L2 regularization (also known as Ridge). L1 regularization adds the absolute value of the theta coefficients to the cost function, while L2 regularization adds the squared value of the theta coefficients to the cost function.

## Contributing
If you have any suggestions or improvements for the linear regression model, feel free to create a pull request or submit an issue.

## Acknowledgements
This implementation was inspired by the Andrew Ng's Machine Learning course on Coursera.

## License
This project is licensed under the MIT License.
