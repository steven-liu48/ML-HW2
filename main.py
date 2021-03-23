# COMS 4771: Homework 2
# Question 4： Understanding model complexity and overfitting
# Steven Liu (xl2948)

from scipy.io import loadmat
import numpy as np


# Load the data from .mat files
def load_data():
    print("="*60)
    print("Loading data...")
    data = loadmat("./digits.mat")

    # numpy.ndarray, (10000, 784)
    X = data['X']

    # numpy.ndarray, (1, 10000)
    y = data['Y']

    return X, y


# Split the data using split rate, return np arrays X_train, X_test, y_train and y_test
def train_test_split(X, y, rate):
    print("="*60)
    print("Splitting data...")
    print("="*60)
    X_y = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
    X2 = X_y[:, :X.size // len(X)].reshape(X.shape)
    y2 = X_y[:, X.size // len(X):].reshape(y.shape)
    np.random.shuffle(X_y)

    train_num = int(len(X) * rate)
    print("Total data:", len(X))
    print("train_num:", train_num)
    print("test_num:", len(X)-train_num)

    X_train = X2[:train_num, :]
    X_test = X2[train_num:, :]
    y_train = y2[:train_num, :]
    y_test = y2[train_num:, :]

    return X_train, X_test, y_train, y_test


# Decision Tree Helper - Return the best split as a dict of feature, left dataset and right dataset
def best_split(dataset, threshold):
    max_entropy = float('inf')
    max_feature = -1
    max_left = None
    max_right = None
    for i in range(784):
        # Split into left and right subsets using feature i
        left, right = [], []
        for data in dataset:
            # If reach the threshold
            if data[0][i] > threshold:
                left.append(data)
            # If not
            else:
                right.append(data)

        # Calculate reduction in uncertainty
        entropy = entropy_of(left, right)
        if entropy < max_entropy:
            max_feature = i
            max_entropy = entropy
            max_left = left
            max_right = right

    return {'feature': max_feature, 'left': max_left, 'right': max_right, 'leaf': False}


# Decision Tree Helper - Get the majority label of a dataset, the max proportion, and count of each label
def get_majority(dataset):
    majority = None
    max_proportion = 0
    counter = [0] * 10

    for data in dataset:
        counter[data[1]] += 1

    for i in range(10):
        proportion = counter[i]/len(dataset)
        if proportion > max_proportion:
            max_proportion = proportion
            majority = i

    return counter, majority, max_proportion


# Decision Tree Helper - Calculate entropy of a split
def entropy_of(left, right):
    entropy = 0

    for dataset in [left, right]:
        # Skip if no entry in the set
        if len(dataset) == 0:
            continue
        counter, majority, majority_prop = get_majority(dataset)
        dataset_entropy = 0
        total_elements = sum(counter)
        for i in range(10):
            prop = counter[i]/total_elements
            if prop == 0:
                continue
            dataset_entropy += prop * np.log(1/prop)
        entropy += len(dataset)/(len(left)+len(right)) * dataset_entropy

    return entropy


# Decision Tree Helper - The main splitting function, using recursion
def split_at_root(root, max_depth, depth, threshold):
    left = root['left']
    right = root['right']
    
    # If a subset has zero elements
    if len(left) == 0 or len(right) == 0:
        _, majority, _ = get_majority(left+right)
        root['left'] = {'feature': majority, 'leaf': True}
        root['right'] = {'feature': majority, 'leaf': True}
        # print("Leaf: Zero elements")
        return

    # If reach max depth
    if depth >= max_depth:
        _, majority, _ = get_majority(left)
        root['left'] = {'feature': majority, 'leaf': True}
        _, majority, _ = get_majority(right)
        root['right'] = {'feature': majority, 'leaf': True}
        # print("Leaf: Max Depth reached")
        return
    
    # Recursion
    root['left'] = best_split(left, threshold)
    split_at_root(root['left'], max_depth, depth+1, threshold)
    root['right'] = best_split(right, threshold)
    split_at_root(root['right'], max_depth, depth+1, threshold)


# Build a tree on training data and return the root
def build_decision_tree(X_train, y_train, K, threshold):
    # print("Building decision tree...")

    # Dataset: a list of tuples [data, tag]
    dataset = []
    for i in range(len(X_train)):
        new_tuple = (X_train[i], y_train[i].item(0))
        dataset.append(new_tuple)

    # Build the tree
    root = best_split(dataset, threshold)  # Split at root
    split_at_root(root, K, 1, threshold)
    return root


# Benchmark
def benchmark(root, dataset, labels, threshold):
    print("="*60)
    print("Benchmarking...")
    # Test
    pred = []
    for data in dataset:
        node = root
        feature = node['feature']
        while node is not None:
            if node['leaf']:
                pred.append(node['feature'])
                break
            if data[feature] > threshold:
                node = node['left']
            else:
                node = node['right']
            feature = node['feature']

    # Calculate error
    correct_num = 0
    error_num = 0
    for i in range(len(labels)):
        if pred[i] == labels[i]:
            correct_num += 1
        else:
            error_num += 1

    return error_num/(error_num + correct_num)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Parameters
    rate_of_training = 0.8
    K = 11

    # Load data
    X, y = load_data()

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, rate_of_training)

    # Run decision tree classifier
    tree = build_decision_tree(X_train, y_train, K, 10)
    training_error = benchmark(tree, X_train, y_train, 10)
    print("Training Error:", training_error)
    testing_error = benchmark(tree, X_test, y_test, 10)
    print("Testing Error:", testing_error)

    # # Code for training/testing error analysis
    # train = []
    # test = []
    #
    # for i in range(30):
    #     X, y = load_data()
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, 0.4)
    #     print("K:", i+1)
    #     tree = build_decision_tree(X_train, y_train, i+1, 10)
    #     training_error = benchmark(tree, X_train, y_train, 10)
    #     testing_error = benchmark(tree, X_test, y_test, 10)
    #     print("training_error:", training_error)
    #     print("testing_error:", testing_error)
    #     train.append(training_error)
    #     test.append(testing_error)
    #
    # print(train)
    # print(test)