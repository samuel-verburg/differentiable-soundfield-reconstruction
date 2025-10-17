"""Differentiable Physics Neural Network (DPNN)."""

import os

import numpy as np
import torch
from siren_pytorch import SirenNet as SIREN
from tqdm import tqdm

from utils import *


class DPNN:
    def __init__(self, L, T, c, num_layers=3, num_hidden=64, device="cpu"):
        self.L = L
        self.T = T
        self.c = c
        self.device = device

        self.nn = SIREN(dim_in=2, dim_hidden=num_hidden, dim_out=1, num_layers=num_layers, w0_initial=30).to(device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=1e-4)

        self.epoch = 0
        self.lambdas = torch.tensor([1, 1], device=device)

        # loss terms: data, sparsity
        self.loss_hist = [[], []]
        self.lamb_hist = [[], []]

    def forward(self, grid):
        """Forward pass to predict initial field p0."""
        xy = grid.xy.clone()
        return self.nn(xy).view(grid.n_L, grid.n_L)

    def compute_field(self, grid):
        """Compute total field from predicted initial field p0."""
        p0 = self.forward(grid)
        cdtdx2 = (self.c * grid.dt / grid.dx) ** 2
        beta = (self.c * grid.dt - grid.dx) / (self.c * grid.dt + grid.dx)

        # 3x3 Laplacian kernel
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=p0.dtype, device=self.device
        ).view(1, 1, 3, 3)

        self.p = _time_integrator(p0, grid.n_T, cdtdx2, kernel, beta, bc_type="absorbing")
        self.p0 = p0

    def loss_data(self, array, grid):
        p = self.p.view(-1, grid.n_T)
        data = array.data.clone()
        return torch.mean((p[array.idx] - data) ** 2)

    def loss_sparse(self):
        return torch.mean(torch.abs(self.p0))

    def closure(self, array, grid, update_rate):
        self.optimizer.zero_grad()
        self.compute_field(grid)

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
    """FDTD solver for 2D wave equation."""
    frames = []
    frames.append(p0)
    timestep = 1
    while timestep < n_T:
        p_prev = frames[-1]
        laplacian = torch.nn.functional.conv2d(p_prev.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)

        if timestep == 1:
            p_next_inner = p_prev[1:-1, 1:-1] + 0.5 * cdtdx2 * laplacian
        else:
            p_prev2 = frames[-2]
            p_next_inner = 2 * p_prev[1:-1, 1:-1] - p_prev2[1:-1, 1:-1] + cdtdx2 * laplacian

        p_next = p_prev.clone()
        p_next[1:-1, 1:-1] = p_next_inner

        # boundary handling
        if bc_type == "dirichlet":
            p_next[0, :] = 0.0
            p_next[-1, :] = 0.0
            p_next[:, 0] = 0.0
            p_next[:, -1] = 0.0
        elif bc_type == "neumann":
            p_next[:, 0] = p_next[:, 1]
            p_next[:, -1] = p_next[:, -2]
            p_next[0, :] = p_next[1, :]
            p_next[-1, :] = p_next[-2, :]
        elif bc_type == "absorbing":
            # first-order absorbing boundary condition
            p_next[:, 0] = p_prev[:, 1] + beta * (p_next[:, 1] - p_prev[:, 0])
            p_next[:, -1] = p_prev[:, -2] + beta * (p_next[:, -2] - p_prev[:, -1])
            p_next[0, :] = p_prev[1, :] + beta * (p_next[1, :] - p_prev[0, :])
            p_next[-1, :] = p_prev[-2, :] + beta * (p_next[-2, :] - p_prev[-1, :])
        else:
            raise ValueError("Unknown bc_type: %s. Supported: dirichlet, neumann, absorbing." % bc_type)

        frames.append(p_next)
        timestep += 1

    return torch.stack(frames, dim=2)