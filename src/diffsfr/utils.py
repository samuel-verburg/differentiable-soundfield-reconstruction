"""Utility helpers used throughout the repository."""


import numpy as np
import torch


class Grid:
    """Grid container."""

    def __init__(self, xx, yy, t, xy):
        self.xx = xx
        self.yy = yy
        self.t = t
        self.xy = xy
        self.n_L = xx.shape[0]
        self.n_T = t.shape[0]
        self.xyt = None
        # grid spacing assumed uniform
        self.dx = xx[0, 1] - xx[0, 0]
        self.dy = yy[1, 0] - yy[0, 0]
        self.dt = t[1] - t[0]

    def downsample(self, factor):
        """Return a downsampled Grid instance by integer `factor`.

        This is used to create a coarser grid for DPNN.
        """
        xx_ds = self.xx[::factor, ::factor]
        yy_ds = self.yy[::factor, ::factor]
        t_ds = self.t[::factor]
        xy_ds = np.column_stack((xx_ds.ravel(), yy_ds.ravel()))
        return Grid(xx_ds, yy_ds, t_ds, xy_ds)


def make_tensor(array, device, dtype=torch.float32):
    """Convert a numpy array to a torch tensor on `device`."""
    return torch.tensor(array.copy(), dtype=dtype, device=device)


def make_xyt_tensor(xy, t, device, dtype=torch.float32):
    """Create an (N_xy * N_t, 3) tensor with repeated (x,y) and tiled t values.

    The returned tensor requires gradients and is used as NN input.
    """
    xy_repeated = np.repeat(xy.copy(), len(t), axis=0)
    t_tiled = np.tile(t.copy(), (len(xy), 1)).flatten()
    xyt = np.column_stack((xy_repeated, t_tiled))
    return torch.tensor(xyt, dtype=dtype, device=device, requires_grad=True)


def scale_domain(L=1.0, T=2e-3, c=343.0):
    """Normalize the domain"""
    T_scaled = T * c / L
    L_scaled = L / L
    c_scaled = 1.0
    return L_scaled, T_scaled, c_scaled


def find_index_nearest_xy(xy_grid, xy_points):
    """Find nearest grid indices for a set of xy points."""
    indices = []
    max_distance = 0.0
    for point in xy_points:
        distances = np.linalg.norm(xy_grid - point, axis=1)
        nearest_index = int(np.argmin(distances))
        indices.append(nearest_index)
        max_distance = max(max_distance, float(distances[nearest_index]))
    return indices, max_distance


def sobol_colloc(num_points, L_min, L_max, T_min, T_max, seed, device):
    """Generate Sobol quasi-random collocation points in (x,y,t)."""
    sobol_engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    xyt = sobol_engine.draw(num_points).to(device)
    xyt[:, :2] = xyt[:, :2] * (L_max - L_min) + L_min
    xyt[:, 2] = xyt[:, 2] * (T_max - T_min) + T_min
    xyt.requires_grad_(True)
    return xyt


def sobol_colloc_boundary(num_points, L_min, L_max, T_min, T_max, boundary_id, seed, device):
    """Generate Sobol points on the boundary side.

    `boundary_id` should be one of ['left'], ['right'], ['top'], ['bottom'].
    """
    sobol_engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    xyt = sobol_engine.draw(num_points).to(device)
    xyt[:, :2] = xyt[:, :2] * (L_max - L_min) + L_min
    xyt[:, 2] = xyt[:, 2] * (T_max - T_min) + T_min

    if boundary_id == ["left"]:
        xyt[:, 0] = L_min
    elif boundary_id == ["right"]:
        xyt[:, 0] = L_max
    elif boundary_id == ["bottom"]:
        xyt[:, 1] = L_min
    elif boundary_id == ["top"]:
        xyt[:, 1] = L_max

    xyt.requires_grad_(True)
    return xyt


def random_colloc(num_points, L_min, L_max, T_min, T_max, device):
    """Uniform random collocation points in (x,y,t)."""
    xyt = torch.rand(num_points, 3, device=device)
    xyt[:, 0] = xyt[:, 0] * (L_max - L_min) + L_min
    xyt[:, 1] = xyt[:, 1] * (L_max - L_min) + L_min
    xyt[:, 2] = xyt[:, 2] * (T_max - T_min) + T_min
    xyt.requires_grad_(True)
    return xyt


def loss_grad_norm(loss, model):
    """Compute the norm of gradients of `loss` w.r.t. model parameters.

    A single backward pass for all parameters (one call per parameter tensor would
    repeat the backward pass, including the solver, for every tensor).
    """
    params = [param for param in model.parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = torch.tensor(0.0, device=loss.device)
    for grad_param in grads:
        if grad_param is None:
            continue
        total = total + torch.sum(grad_param ** 2)
    grad_norm = total ** 0.5
    return grad_norm.detach()

def update_lambda(model, losses, lambdas, alpha=0.9, max_lamb=1e6):
    """Adaptive loss-weight update used by PINN/DPNN training loops.

    Returns a list of updated lambda tensors (one per loss term).
    """
    grads = [loss_grad_norm(loss_item, model) for loss_item in losses]
    grad_sum = sum([g for g in grads])
    new_lambdas = []
    for i, g in enumerate(grads):
        lamb_hat = grad_sum / g if g != 0 else torch.ones_like(g)
        if torch.isnan(lamb_hat) or torch.isinf(lamb_hat):
            lamb_hat = torch.ones_like(lamb_hat)
        lamb_updated = min(alpha * lambdas[i] + (1 - alpha) * lamb_hat, torch.tensor(max_lamb))
        new_lambdas.append(lamb_updated)
    return new_lambdas


def log_history(loss_history, losses, lamb_history, lambdas):
    """Append current loss and lambda values into the history lists."""
    for i in range(len(losses)):
        loss_history[i].append(losses[i].item())
        lamb_history[i].append(float(lambdas[i]))


def compute_total_loss(losses, lambdas):
    """Compute the weighted total loss used in optimizer steps."""
    total_loss = losses[0]
    for i in range(1, len(losses)):
        total_loss = total_loss + losses[i] * lambdas[i] / lambdas[0]
    return total_loss


def mse(p_true, p_est, axis):
    return float(np.mean((p_true - p_est) ** 2, axis=axis))


def nmse(p_true, p_est, axis):
    return float(np.mean((p_true - p_est) ** 2, axis=axis) / np.mean(p_true ** 2, axis=axis))
