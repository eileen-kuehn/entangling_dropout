from typing import Tuple
import numpy as np


def generate_data(
    samples: int = 15, noise: float = 1, seed: int = 1337
) -> Tuple[str, np.ndarray]:
    rd = np.random.RandomState(seed)
    x_values = np.linspace(0, 2, samples) * np.pi
    distance = x_values[1] - x_values[0]
    x_noise = (rd.random_sample(size=samples) - 0.5) * distance * noise
    noisy_x_values = x_values + x_noise
    y_values = np.sin(x_values)
    return noisy_x_values, y_values
