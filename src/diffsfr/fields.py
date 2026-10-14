"""The SoundField class is a container for the sound fields used in training the PINN and DPNN. 

It includes routines for generating and saving synthetic sound fields used in the examples. 
The "exact" solution is computed using `scipy.integrate.quad_vec` which might take a long time for large grids. 
Consider using a faster numerical solver for generating data.
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.integrate import quad_vec
from scipy.special import jv

from diffsfr.sensors import pseudo_random_array
from diffsfr.utils import scale_domain, Grid


# Module names under which older pickles stored the package classes, e.g. the example
# data files reference `utils.Grid` and `__main__.SoundField`.
_LEGACY_MODULES = {"utils": "diffsfr.utils", "src.utils": "diffsfr.utils",
                   "fields": "diffsfr.fields", "src.fields": "diffsfr.fields"}
_MAIN_CLASSES = {"SoundField": "diffsfr.fields", "Grid": "diffsfr.utils"}


class _SoundFieldUnpickler(pickle.Unpickler):
    """Resolve module names used by pre-package example data files."""

    def find_class(self, module, name):
        if module == "__main__" and name in _MAIN_CLASSES:
            module = _MAIN_CLASSES[name]
        module = _LEGACY_MODULES.get(module, module)
        return super().find_class(module, name)


class SoundField:
    """Container for sound fields."""

    def __init__(self, L, T, c, ic_type, grid):
        self.L = L
        self.T = T
        self.c = c
        self.ic_type = ic_type
        self.grid = grid
        self.p = np.array([])

    def compute_field(self):
        """Compute the pressure field based on initial condition."""
        if self.ic_type == "pulse":
            sigma = 2e-2
            center_x, center_y = self.L / 2.0, self.L / 2.0
            amplitude = 1.0
            r = np.sqrt((self.grid.xy[:, 0] - center_x) ** 2 + (self.grid.xy[:, 1] - center_y) ** 2)
            field = _integrate_sources(r, self.grid.t, sigma, amplitude)
            self.p = field.reshape((self.grid.n_L, self.grid.n_L, self.grid.n_T))

        elif self.ic_type == "twenty_pulses":
            sigma = 2e-2
            num_sources = 20
            source_positions = pseudo_random_array(num_sources, L_min=self.L * 0.3, L_max=self.L * 0.7, min_distance=0.05)
            amplitudes = np.random.uniform(-1, 1, num_sources)
            field = np.zeros((self.grid.n_L ** 2, self.grid.n_T))
            for src_idx in range(num_sources):
                r = np.sqrt((self.grid.xy[:, 0] - source_positions[src_idx, 0]) ** 2 + (self.grid.xy[:, 1] - source_positions[src_idx, 1]) ** 2)
                field += _integrate_sources(r, self.grid.t, sigma, amplitudes[src_idx])
            self.p = field.reshape((self.grid.n_L, self.grid.n_L, self.grid.n_T))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{self.ic_type}.pkl"), "wb") as fh:
            pickle.dump(self, fh)

    def load(self, path):
        with open(os.path.join(path, f"{self.ic_type}.pkl"), "rb") as fh:
            return _SoundFieldUnpickler(fh).load()


def integrand(xi, r, t, sigma):
    return np.exp(-((xi * sigma) ** 2) / 2.0) * np.cos(xi * t) * jv(0, xi * r) * xi


def _integrate_sources(r, t, sigma, amplitude):
    field = np.zeros((len(r), len(t)))
    print("Computing reference field using quad_vec... (might be slow)")
    for k in tqdm(range(len(t))):
        integral_values, _ = quad_vec(integrand, 0, np.inf, args=(r, t[k], sigma), workers=-1)
        field[:, k] = integral_values * amplitude * sigma ** 2
    return field


if __name__ == "__main__":
    """Generate data for examples."""
    L, T, c = scale_domain(L=1, T=1e-3, c=343)
    n_L = 199
    n_T = 99
    x = np.linspace(0, L, n_L)
    y = np.linspace(0, L, n_L)
    t = np.linspace(0, T, n_T)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    grid = Grid(xx, yy, t, xy)
    for ic_type in ["pulse", "twenty_pulses"]:
        soundfield = SoundField(L, T, c, ic_type=ic_type, grid=grid)
        soundfield.compute_field()
        soundfield.save("../examples/data")