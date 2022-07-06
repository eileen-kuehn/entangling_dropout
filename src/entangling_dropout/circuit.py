import pennylane as qml
import pennylane.numpy as np


def encoding_circuit(data: float, qubits: int = 5):
    assert (
        data >= -1 and data <= 1
    ), f"data must be in the range [-1, 1], but is at {data}"
    for qubit in range(qubits):
        qml.RY(np.arcsin(data), qubit)
        qml.RY(np.arccos(data**2), qubit)


def tuning_circuit(params: np.ndarray, qubits: int = 5):
    gates = [qml.RX, qml.RZ, qml.RX]
    for i in range(len(gates)):
        for qubit in range(qubits):
            gates[i](params[qubit][i], qubit)
            if qubit < qubits - 1:
                qml.CNOT(wires=[qubit, qubit + 1])


def circuit(data: float, params: np.ndarray, layers: int, qubits: int):
    for layer in range(layers):
        encoding_circuit(data, qubits)
        tuning_circuit(params[layer], qubits)
    return [qml.expval(qml.PauliZ(0))]
