"""Sensor array container and related utilities:
get sensor locations, extract data from reference fields, and add noise.
"""

import numpy as np
from scipy.signal import decimate

from diffsfr.utils import find_index_nearest_xy


class SensorArray:
    """Container for a sensor array."""

    def __init__(self, n, distribution, L_min, L_max, min_distance, snr):
        self.n = n
        self.distribution = distribution
        self.L_min = L_min
        self.L_max = L_max
        self.min_distance = min_distance
        self.snr = snr

        self.xy = np.array([])
        self.t = np.array([])
        self.idx = []
        self.data = np.array([])

    def get_positions(self, grid):
        """Sample sensor positions according to `distribution` and snap them to grid points."""
        if self.distribution == "uniform":
            self.xy = uniform_array(self.n, self.L_min, self.L_max)
        elif self.distribution == "random":
            self.xy = random_array(self.n, self.L_min, self.L_max)
        elif self.distribution == "pseudo_random":
            self.xy = pseudo_random_array(self.n, self.L_min, self.L_max, self.min_distance)
        else:
            raise ValueError('Unknown distribution type. Choose from "uniform", "random", "pseudo_random".')

        # snap to nearest grid locations and save indices
        indices, _ = find_index_nearest_xy(grid.xy, self.xy)
        self.xy = grid.xy[indices, :]
        self.t = grid.t
        self.idx = indices

    def get_data(self, soundfield, factor):
        """Extract and downsample reference field data at sensor locations.

        `factor` performs decimation along the time axis to match DPNN grid.
        """
        indices, _ = find_index_nearest_xy(soundfield.grid.xy, self.xy)
        reshaped_field = soundfield.p.reshape(-1, soundfield.p.shape[-1]).copy()
        self.data = reshaped_field[indices, :]
        # temporal downsampling
        self.data = decimate(self.data, factor, axis=1)
        # optionally add white noise
        if self.snr is not None:
            self.data = add_noise(self.data, self.snr)


def uniform_array(n, L_min, L_max):
    sqrt_n = int(np.sqrt(n))
    xx, yy = np.meshgrid(
        np.linspace(L_min, L_max, sqrt_n), np.linspace(L_min, L_max, sqrt_n)
    )
    return np.column_stack((xx.ravel(), yy.ravel()))


def random_array(n, L_min, L_max, seed=21):
    np.random.seed(seed)
    return np.random.uniform(L_min, L_max, (n, 2))


def pseudo_random_array(n, L_min, L_max, min_distance=0.025, seed=21):
    """Greedy point sampling with a minimal separation constraint."""
    np.random.seed(seed)
    points = []
    attempts = 0
    max_attempts = 1000 * n
    while len(points) < n and attempts < max_attempts:
        candidate = np.random.uniform(L_min, L_max, 2)
        if not points:
            points.append(candidate)
        else:
            distances = np.linalg.norm(np.array(points) - candidate, axis=1)
            if np.min(distances) > min_distance:
                points.append(candidate)
        attempts += 1
    if len(points) < n:
        raise ValueError("Could not find enough valid points within the maximum number of attempts")
    return np.array(points)


def add_noise(data, snr, quiet=True, seed=1):
    """Add Gaussian noise to `data` to achieve target SNR (in dB)."""
    np.random.seed(seed)
    data_power = np.sum(data ** 2) / data.size
    data_power_db = 10 * np.log10(data_power)
    noise_power_db = data_power_db - snr
    noise_power = 10 ** (noise_power_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), size=data.shape)
    if not quiet:
        actual_snr = 10 * np.log10(np.sum(data ** 2) / np.sum(noise ** 2))
        print("Actual SNR:", actual_snr, "dB")
    data_with_noise = data + noise
    return data_with_noise
