# Gradient Descent Linear Regression Model from Scratch

This repository contains a simple implementation of a linear regression model using the gradient descent algorithm with L2 regularization, built from scratch in Python without the use of external libraries. The model is designed to take in a set of input features and target values, and outputs a set of slope and intercept coefficients that can be used to predict new target values from new input features.

## Installation

To use this model, simply clone the repository to your local machine:

```sh
git clone https://github.com/<username>/<repository-name>.git


## How to Use
To use the linear regression model, you can import the fit_linear_regression and predict_linear_regression functions from the linear_regression.py file:

<code>from linear_regression import fit_linear_regression, predict_linear_regression</code>

The fit_linear_regression function takes in an array of input features (X), an array of target values (y), the learning rate (alpha), the regularization parameter (lambda_), and the number of iterations to run the gradient descent algorithm (num_iterations). It returns a tuple containing the slope and intercept coefficients of the linear regression model.

The predict_linear_regression function takes in the slope and intercept coefficients (theta_0 and theta_1, respectively) and an array of input values (X), and returns an array of predicted output values.

## Modifying the Model
You can modify the existing implementation to potentially improve the accuracy of the linear regression model. Some possible modifications include:

+ Feature scaling: You can scale the input features to have a similar range, which can help the gradient descent algorithm converge more quickly. To do this, you can subtract the mean of each feature and divide by the standard deviation.
+ Regularization: You can add a regularization term to the cost function to prevent overfitting. There are two commonly used types of regularization: L1 regularization (also known as Lasso) and L2 regularization (also known as Ridge). L1 regularization adds the absolute value of the theta coefficients to the cost function, while L2 regularization adds the squared value of the theta coefficients to the cost function.

## Contributing
If you have any suggestions or improvements for the linear regression model, feel free to create a pull request or submit an issue.

## License
This project is licensed under the MIT License.
