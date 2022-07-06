import pennylane.numpy as np


def mse_cost(params, x_values, y_values, qnode):
    predictions = []
    for x_value in x_values:
        predictions.extend(qnode(x_value, params))
    return np.sum((np.array(predictions) - y_values) ** 2) / len(y_values)
