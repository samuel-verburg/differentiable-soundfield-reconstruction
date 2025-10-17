"""Physics-Informed Neural Network (PINN)."""

import os

import numpy as np
import torch
from siren_pytorch import SirenNet as SIREN
from torch.autograd import grad
from tqdm import tqdm

from utils import sobol_colloc, sobol_colloc_boundary, make_xyt_tensor, compute_total_loss, update_lambda, log_history


class PINN:
    def __init__(self, L, T, c, num_layers=4, num_hidden=128, device="cpu"):
        self.L = L
        self.T = T
        self.c = c
        self.device = device

        self.nn = SIREN(dim_in=3, dim_hidden=num_hidden, dim_out=1, num_layers=num_layers, w0_initial=30).to(device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=1e-4)

        self.epoch = 0
        self.lambdas = torch.tensor([1, 1, 1, 1], device=device)

        # loss terms: data, pde, sparsity, boundary
        self.loss_hist = [[], [], [], []]
        self.lamb_hist = [[], [], [], []]

    def loss_data(self, array):
        xyt = array.xyt.clone()
        data = array.data.clone()
        p_pred = self.nn(xyt).view(-1, data.shape[-1])
        return torch.mean((p_pred - data) ** 2)

    def loss_pde(self):
        collocation_points = sobol_colloc(int(2e3), 0, self.L, 0, self.T, seed=None, device=self.device)
        p = self.nn(collocation_points)
        p_grad = grad(p, collocation_points, torch.ones_like(p), create_graph=True)[0]
        p_x, p_y, p_t = p_grad[:, 0], p_grad[:, 1], p_grad[:, 2]
        p_xx = grad(p_x, collocation_points, torch.ones_like(p_x), create_graph=True)[0][:, 0]
        p_yy = grad(p_y, collocation_points, torch.ones_like(p_y), create_graph=True)[0][:, 1]
        p_tt = grad(p_t, collocation_points, torch.ones_like(p_t), create_graph=True)[0][:, 2]
        wave_eq_residual = p_xx + p_yy - p_tt / (self.c ** 2)
        return torch.mean(wave_eq_residual ** 2)

    def loss_sparse(self):
        collocation_points = sobol_colloc(int(2e2), 0, self.L, 0, self.T * 0.1, seed=None, device=self.device)
        p0 = self.nn(collocation_points)
        return torch.mean(torch.abs(p0))

    def loss_boundary(self):
        total_residual = torch.tensor(0.0, device=self.device)
        for boundary_id in ["left", "right", "bottom", "top"]:
            xyt = sobol_colloc_boundary(int(2e2), 0, self.L, 0, self.T, [boundary_id], seed=None, device=self.device)
            p = self.nn(xyt)
            p_grad = grad(p, xyt, torch.ones_like(p), create_graph=True)[0]
            p_x = p_grad[:, 0]
            p_y = p_grad[:, 1]
            p_t = p_grad[:, 2]

            if boundary_id == "left":
                residual = p_x + p_t / self.c
            elif boundary_id == "right":
                residual = p_x - p_t / self.c
            elif boundary_id == "bottom":
                residual = p_y + p_t / self.c
            elif boundary_id == "top":
                residual = p_y - p_t / self.c

            total_residual = total_residual + torch.mean(residual ** 2)
        return total_residual

    def closure(self, array, update_rate):
        self.optimizer.zero_grad()

        losses = [self.loss_data(array), self.loss_pde(), self.loss_sparse(), self.loss_boundary()]

        if self.epoch % update_rate == 0:
            self.lambdas = update_lambda(self.nn, losses, self.lambdas)

        total_loss = compute_total_loss(losses, self.lambdas)
        total_loss.backward()

        log_history(self.loss_hist, losses, self.lamb_hist, self.lambdas)

        self.epoch += 1
        if self.epoch % 1000 == 999:
            print(f"[{self.epoch + 1:5d}] loss: {total_loss.item()*1e3:.3f}*1e-3")

    def train(self, array, n_epochs=1, update_rate=100):
        for _ in tqdm(range(n_epochs)):
            self.closure(array, update_rate=update_rate)
            self.optimizer.step()

    def compute_field(self, grid, split=True):
        """Evaluate the trained network on the provided grid and store `self.p`.

        For large grids use `split=True` to evaluate one time slice at a time
        which reduces peak GPU memory usage.
        """
        if split:
            p_full = np.zeros((grid.n_L, grid.n_L, grid.n_T))
            for i in range(grid.n_T):
                xyt = make_xyt_tensor(grid.xy, grid.t[i : i + 1], self.device)
                p_slice = self.nn(xyt)
                p_slice = p_slice.view(grid.n_L, grid.n_L).detach().cpu().numpy()
                p_full[:, :, i] = p_slice
            self.p = p_full
        else:
            p = self.nn(grid.xyt)
            self.p = p.reshape(grid.n_L, grid.n_L, grid.n_T)

    def save_nn(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.nn.state_dict(), os.path.join(path, "pinn.pt"))
        np.save(os.path.join(path, "pinn_loss_hist.npy"), self.loss_hist)
        np.save(os.path.join(path, "pinn_lamb_hist.npy"), self.lamb_hist)

    def load_nn(self, path, device=None):
        map_location = device if device is not None else "cpu"
        self.nn.load_state_dict(torch.load(os.path.join(path, "pinn.pt"), map_location=map_location))
        self.nn.eval()
        self.nn.to(map_location)
        self.loss_hist = np.load(os.path.join(path, "pinn_loss_hist.npy"), allow_pickle=True)
        self.lamb_hist = np.load(os.path.join(path, "pinn_lamb_hist.npy"), allow_pickle=True)
        return self.nn