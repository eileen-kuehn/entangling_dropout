from enum import Enum
import logging
from typing import Tuple, Union
import numpy as np


class NoiseType(Enum):
    NONE = 0
    TIME_ERROR = 1
    AMPLITUDE_ERROR = 2


def convert_noise_type(noise_type: Union[NoiseType, str, int]) -> NoiseType:
    if isinstance(noise_type, NoiseType):
        return noise_type
    elif isinstance(noise_type, int):
        try:
            return NoiseType(noise_type)
        except ValueError:
            pass
    elif isinstance(noise_type, str):
        try:
            return getattr(NoiseType, noise_type)
        except AttributeError:
            pass
    logging.warning(f"Received unknown noise type {noise_type}")
    return NoiseType.NONE


def generate_data(
    samples: int = 15,
    noise: float = 1,
    seed: int = 1337,
    noise_type: NoiseType = NoiseType.TIME_ERROR,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    rd = np.random.default_rng(seed)
    x_values = np.linspace(-1, 1, samples)
    y_values = np.sin(x_values * np.pi)
    # add noise
    scale_x = scale_y = 1.0
    if noise_type == NoiseType.TIME_ERROR:
        distance = np.abs(x_values[1] - x_values[0])
        x_noise = rd.normal(loc=0, scale=distance * noise, size=samples)
        x_values = x_values + x_noise
        scale_x = max(max(x_values), abs(min(x_values)))
    elif noise_type == NoiseType.AMPLITUDE_ERROR:
        y_noise = rd.normal(loc=0, scale=noise, size=samples)
        y_values = y_values + y_noise
        scale_y = max(max(y_values), abs(min(y_values)))
    else:
        logging.warning(f"Not introducing noise for type {noise_type}")
    return x_values / scale_x, y_values / scale_y, scale_x, scale_y
