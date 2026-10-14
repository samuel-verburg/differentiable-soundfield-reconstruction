"""Differentiable Physics Neural Network (DPNN)."""

import os

import numpy as np
import torch
from siren_pytorch import SirenNet as SIREN
from tqdm import tqdm

from diffsfr.utils import compute_total_loss, log_history, update_lambda
from diffsfr.wave_solver import (
    WaveSensorOp,
    build_sensor_operator,
    laplacian_kernel,
    solver_coefficients,
    time_integrator,
)

# How the gradient of the data loss is computed through the solver:
#   "adjoint"  - hand-written discrete adjoint (WaveSensorOp); memory independent of n_T
#   "operator" - precomputed sensor operator G (one matrix-vector product per iteration);
#                memory M_ob * n_T * n_L**2, best for small 2D problems
#   "autograd" - PyTorch autodiff through every solver step (original implementation)
GRADIENT_MODES = ("adjoint", "operator", "autograd")


class DPNN:
    def __init__(self, L, T, c, num_layers=3, num_hidden=64, device="cpu", gradient="adjoint", bc_type="absorbing"):
        if gradient not in GRADIENT_MODES:
            raise ValueError("Unknown gradient mode: %s. Supported: %s." % (gradient, ", ".join(GRADIENT_MODES)))
        self.L = L
        self.T = T
        self.c = c
        self.device = device
        self.gradient = gradient
        self.bc_type = bc_type

        self.nn = SIREN(dim_in=2, dim_hidden=num_hidden, dim_out=1, num_layers=num_layers, w0_initial=30).to(device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=1e-4)

        self.epoch = 0
        self.lambdas = torch.tensor([1, 1], device=device)

        # loss terms: data, sparsity
        self.loss_hist = [[], []]
        self.lamb_hist = [[], []]

        # set by prepare(): sensor indices and, for gradient="operator", the sensor operator
        self._idx = None
        self._G = None

    def forward(self, grid):
        """Forward pass to predict initial field p0."""
        xy = grid.xy.clone()
        return self.nn(xy).view(grid.n_L, grid.n_L)

    def _solver_setup(self, grid, p0):
        k2, beta = solver_coefficients(self.c, grid.dt, grid.dx)
        kernel = laplacian_kernel(p0.dtype, p0.device)
        return k2, beta, kernel

    def compute_field(self, grid, requires_grad=False):
        """Compute total field from predicted initial field p0 (stored in self.p, shape (n_L, n_L, n_T)).

        Used for inference. Training uses compute_traces(), which only evaluates the
        field at the sensors. Pass requires_grad=True to differentiate through the
        full field with autograd.
        """
        with torch.set_grad_enabled(requires_grad):
            p0 = self.forward(grid)
            k2, beta, kernel = self._solver_setup(grid, p0)
            self.p = time_integrator(p0, grid.n_T, k2, kernel, beta, bc_type=self.bc_type)
            self.p0 = p0

    def prepare(self, array, grid):
        """Cache sensor indices (and the sensor operator for gradient="operator") for training."""
        self._idx = torch.as_tensor(np.asarray(array.idx), dtype=torch.long, device=self.device)
        self._G = None
        if self.gradient == "operator":
            dtype = next(self.nn.parameters()).dtype
            k2, beta = solver_coefficients(self.c, grid.dt, grid.dx)
            kernel = laplacian_kernel(dtype, self._idx.device)
            self._G = build_sensor_operator(
                self._idx, (grid.n_L, grid.n_L), grid.n_T, k2, beta, kernel, bc_type=self.bc_type
            )

    def compute_traces(self, grid):
        """Predict p0 and the pressure at the sensors, self.traces of shape (M_ob, n_T)."""
        p0 = self.forward(grid)
        self.p0 = p0
        if self.gradient == "adjoint":
            k2, beta, kernel = self._solver_setup(grid, p0)
            self.traces = WaveSensorOp.apply(p0, self._idx, grid.n_T, k2, beta, kernel, self.bc_type)
        elif self.gradient == "operator":
            self.traces = (self._G @ p0.reshape(-1)).view(-1, grid.n_T)
        else:
            k2, beta, kernel = self._solver_setup(grid, p0)
            p = time_integrator(p0, grid.n_T, k2, kernel, beta, bc_type=self.bc_type)
            self.traces = p.reshape(-1, grid.n_T)[self._idx]

    def loss_data(self, array, grid):
        return torch.mean((self.traces - array.data) ** 2)

    def loss_sparse(self):
        return torch.mean(torch.abs(self.p0))

    def closure(self, array, grid, update_rate):
        if self._idx is None:
            self.prepare(array, grid)
        self.optimizer.zero_grad()
        self.compute_traces(grid)

        losses = [self.loss_data(array, grid), self.loss_sparse()]

        if self.epoch % update_rate == 0:
            self.lambdas = update_lambda(self.nn, losses, self.lambdas)

        total_loss = compute_total_loss(losses, self.lambdas)
        total_loss.backward()

        log_history(self.loss_hist, losses, self.lamb_hist, self.lambdas)

        self.epoch += 1
        if self.epoch % 1000 == 999:
            print(f"[{self.epoch + 1:5d}] loss: {total_loss.item()*1e3:.3f}*1e-3")

    def train(self, array, grid, n_epochs=1, update_rate=100):

        # Stability check for the numerical integrator
        stability_limit = (1 / self.c) * (1 / (1 / grid.dx ** 2 + 1 / grid.dy ** 2) ** 0.5)
        if grid.dt > stability_limit:
            raise ValueError("error: dt=%g exceeds the stability limit %g" % (grid.dt, stability_limit))

        self.prepare(array, grid)
        for _ in tqdm(range(n_epochs)):
            self.closure(array, grid, update_rate=update_rate)
            self.optimizer.step()

    def save_nn(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.nn.state_dict(), os.path.join(path, "dpnn.pt"))
        np.save(os.path.join(path, "dpnn_loss_hist.npy"), self.loss_hist)
        np.save(os.path.join(path, "dpnn_lamb_hist.npy"), self.lamb_hist)

    def load_nn(self, path, device=None):
        map_location = device if device is not None else "cpu"
        self.nn.load_state_dict(torch.load(os.path.join(path, "dpnn.pt"), map_location=map_location))
        self.nn.eval()
        self.nn.to(map_location)
        self.loss_hist = np.load(os.path.join(path, "dpnn_loss_hist.npy"), allow_pickle=True)
        self.lamb_hist = np.load(os.path.join(path, "dpnn_lamb_hist.npy"), allow_pickle=True)
        return self.nn


def _time_integrator(p0, n_T, cdtdx2, kernel, beta, bc_type="absorbing"):
    """FDTD solver for 2D wave equation (kept for backward compatibility; see wave_solver.time_integrator)."""
    return time_integrator(p0, n_T, cdtdx2, kernel, beta, bc_type=bc_type)
