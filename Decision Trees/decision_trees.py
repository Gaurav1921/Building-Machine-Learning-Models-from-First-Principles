import numpy as np

def entropy(y):
    """
    Calculates the entropy of an array of labels.
    
    Parameters:
    - y (array-like): An array of shape (n_samples,) containing the target labels.
    
    Returns:
    - float: The entropy of the input labels.
    """
    n_samples = len(y)
    if n_samples == 0:
        return 0
    n_classes = len(np.unique(y))
    if n_classes <= 1:
        return 0
    counts = np.bincount(y)
    probabilities = counts / n_samples
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def information_gain(X, y, feature_idx, split_value):
    """
    Calculates the information gain of splitting the data based on a given feature and split value.
    
    Parameters:
    - X (array-like): An array of shape (n_samples, n_features) containing the input features.
    - y (array-like): An array of shape (n_samples,) containing the target labels.
    - feature_idx (int): The index of the feature to split on.
    - split_value (float): The value to split the feature on.
    
    Returns:
    - float: The information gain of splitting the data based on the given feature and split value.
    """
    parent_entropy = entropy(y)
    left_mask = X[:, feature_idx] <= split_value
    right_mask = X[:, feature_idx] > split_value
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    n_samples = len(y)
    left_weight = np.sum(left_mask) / n_samples
    right_weight = np.sum(right_mask) / n_samples
    information_gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
    return information_gain

def find_best_split(X, y):
    """
    Finds the best feature and split value to split the data based on information gain.
    
    Parameters:
    - X (array-like): An array of shape (n_samples, n_features) containing the input features.
    - y (array-like): An array of shape (n_samples,) containing the target labels.
    
    Returns:
    - dict: A dictionary containing the best feature index, best split value, left indices, and right indices.
    """
    n_samples, n_features = X.shape
    best_feature_idx, best_split_value, best_information_gain = None, None, -1
    for feature_idx in range(n_features):
        for split_value in np.unique(X[:, feature_idx]):
            ig = information_gain(X, y, feature_idx, split_value)
            if ig > best_information_gain:
                best_feature_idx = feature_idx
                best_split_value = split_value
                best_information_gain = ig
    left_mask = X[:, best_feature_idx] <= best_split_value
    right_mask = X[:, best_feature_idx] > best_split_value
    left_indices = np.where(left_mask)[0]
    right_indices = np.where(right_mask)[0]
    return {'feature_idx': best_feature_idx, 'split_value': best_split_value,
            'left_indices': left_indices, 'right_indices': right_indices}

def decision_tree(X, y, max_depth=2):
    """
    Builds a decision tree for the input data using recursive binary splitting.
    
    Parameters:
    - X (array-like): An array of shape (n_samples, n_features) containing the input features.
    - y (array-like): An array of shape
    - (n_samples,) containing the target labels.
    - max_depth (int): The maximum depth of the decision tree.
    
    Returns:
    - dict: A dictionary representing the decision tree.
    """
    n_samples, n_features = X.shape
    if max_depth == 0 or len(np.unique(y)) == 1:
        leaf_value = np.mean(y)
        return {'leaf_value': leaf_value}
    split = find_best_split(X, y)
    left_tree = decision_tree(X[split['left_indices']], y[split['left_indices']], max_depth-1)
    right_tree = decision_tree(X[split['right_indices']], y[split['right_indices']], max_depth-1)
    return {'feature_idx': split['feature_idx'], 'split_value': split['split_value'],
            'left_tree': left_tree, 'right_tree': right_tree}

def predict_example(x, tree):
    """
    Predicts the target label of a single example using a decision tree.

    Parameters:
    - x (array-like): An array of shape (n_features,) containing a single input example.
    - tree (dict): A dictionary representing the decision tree.
    
    Returns:
    - float: The predicted target label of the input example.
    """
    if 'leaf_value' in tree:
        return tree['leaf_value']
    feature_value = x[tree['feature_idx']]
    if feature_value <= tree['split_value']:
        return predict_example(x, tree['left_tree'])
    else:
        return predict_example(x, tree['right_tree'])

def predict(X, tree):
    """
    Predicts the target labels of an array of input examples using a decision tree.
    
    Parameters:
    - X (array-like): An array of shape (n_samples, n_features) containing the input examples.
    - tree (dict): A dictionary representing the decision tree.
    
    Returns:
    - array-like: An array of shape (n_samples,) containing the predicted target labels.
    """
    return np.array([predict_example(x, tree) for x in X])
