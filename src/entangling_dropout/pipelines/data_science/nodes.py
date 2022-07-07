from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np

from entangling_dropout.circuit import circuit
from entangling_dropout.cost import mse_cost
from entangling_dropout.utils import batch_generator


def get_device(wires: int = 5, shots: int = 1024):
    return qml.device("default.qubit", wires=wires, shots=shots)


def create_cost_fn(qnode):
    return lambda params, x_values, y_values: mse_cost(
        params, x_values, y_values, qnode
    )


def create_qnode(wires: int, layers: int, shots: int):
    dev = get_device(wires=wires, shots=shots)
    return qml.QNode(
        lambda x_value, params: circuit(
            data=x_value, params=params, layers=layers, qubits=wires
        ),
        dev,
    )


def train(
    qnode,
    cost_fn,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    wires: int = 5,
    layers: int = 1,
    stepsize: float = 0.01,
    epochs: int = 1,
    batch_size: int = 1,
    seed: int = 1337,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    rd = np.random.default_rng(seed)
    optimizer = qml.AdamOptimizer(stepsize=stepsize)
    params = rd.uniform(low=-np.pi, high=np.pi, size=(layers, wires, 3))
    history = {
        "loss": [evaluate(cost_fn, params, x_values, y_values)],
        "val_loss": [evaluate(cost_fn, params, x_validation, y_validation)],
    }

    for epoch in range(epochs):
        for batch_x, batch_y in batch_generator(x_values, y_values, batch_size, rd):
            params = optimizer.step(
                lambda weights: mse_cost(weights, batch_x, batch_y, qnode), params
            )

        history["val_loss"].append(
            evaluate(cost_fn, params, x_validation, y_validation)
        )
        history["loss"].append(evaluate(cost_fn, params, x_values, y_values))
        print(
            f"{epoch} Loss: {history['loss'][-1]}, Val loss: {history['val_loss'][-1]}"
        )
        if epoch % 3 == 0:
            stepsize /= 2
            optimizer.stepsize = stepsize
    return params, history


def evaluate(cost_fn, params, x_values, y_values) -> float:
    cost = 0
    for datum, target in zip(x_values, y_values):
        cost += cost_fn(params, [datum], [target])
    return cost / len(y_values)


def generate_loss_curve(
    history: Dict[str, List[float]], loss_test: Optional[float] = None
):
    loss_train = history["loss"]
    loss_val = history["val_loss"]
    epochs = range(0, len(list(history.values())[0]))
    plt.plot(epochs, loss_train, "g", label="Training loss")
    plt.plot(epochs, loss_val, "b", label="Validation loss")
    if loss_test:
        plt.plot(len(list(history.values())[0]) - 1, loss_test, "r+", label="Test loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    return plt


# TODO: create training with entangling dropout
