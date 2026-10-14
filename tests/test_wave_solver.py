"""Tests for the discrete adjoint of the wave solver.

Run with `python -m pytest` from the repository root, or with
`python tests/test_wave_solver.py` from anywhere once the package is installed
(`pip install -e .`). Use the interpreter of the environment that has the
dependencies installed.
"""

import numpy as np
import torch

from diffsfr.wave_solver import (
    BC_TYPES,
    WaveSensorOp,
    _step,
    _step_T,
    adjoint_gradient,
    build_sensor_operator,
    laplacian_kernel,
    record_traces,
    solver_coefficients,
    time_integrator,
)

DTYPE = torch.float64


def _setup(n=12, seed=0):
    torch.manual_seed(seed)
    dx = 1.0 / (n - 1)
    dt = 0.6 * dx  # below the 2D stability limit dx / sqrt(2)
    k2, beta = solver_coefficients(1.0, dt, dx)
    kernel = laplacian_kernel(DTYPE, "cpu")
    return n, k2, beta, kernel


def _inner(a, b):
    return float((a * b).sum())


def test_step_transpose_dot_product():
    """<S x, y> = <x, S^T y> for every boundary condition, general and first step."""
    n, k2, beta, kernel = _setup()
    for bc in BC_TYPES:
        x1, x2, y = (torch.randn(n, n, dtype=DTYPE) for _ in range(3))
        lhs = _inner(_step(x1, x2, k2, beta, kernel, bc, first=False), y)
        aT, bT = _step_T(y, k2, beta, kernel, bc, first=False)
        rhs = _inner(x1, aT) + _inner(x2, bT)
        assert abs(lhs - rhs) <= 1e-12 * max(1.0, abs(lhs)), (bc, lhs, rhs)

        lhs = _inner(_step(x1, None, k2, beta, kernel, bc, first=True), y)
        cT, _ = _step_T(y, k2, beta, kernel, bc, first=True)
        assert abs(lhs - _inner(x1, cT)) <= 1e-12 * max(1.0, abs(lhs)), bc


def test_traces_match_full_field():
    n, k2, beta, kernel = _setup()
    idx = torch.randperm(n * n)[:6]
    for bc in BC_TYPES:
        p0 = torch.randn(n, n, dtype=DTYPE)
        full = time_integrator(p0, 9, k2, kernel, beta, bc_type=bc)
        traces = record_traces(p0, idx, 9, k2, beta, kernel, bc_type=bc)
        assert torch.allclose(traces, full.reshape(-1, 9)[idx], rtol=0, atol=1e-12), bc


def test_adjoint_matches_autograd():
    """Hand-written adjoint gradient equals autograd through the full solver."""
    n, k2, beta, kernel = _setup(n=16)
    idx = torch.randperm(n * n)[:5]
    for bc in BC_TYPES:
        for n_T in (1, 2, 3, 12):
            data = torch.randn(len(idx), n_T, dtype=DTYPE)

            p0 = torch.randn(n, n, dtype=DTYPE, requires_grad=True)
            traces = time_integrator(p0, n_T, k2, kernel, beta, bc_type=bc).reshape(-1, n_T)[idx]
            torch.mean((traces - data) ** 2).backward()
            g_ref = p0.grad.clone()

            p0.grad = None
            traces = WaveSensorOp.apply(p0, idx, n_T, k2, beta, kernel, bc)
            torch.mean((traces - data) ** 2).backward()
            g_adj = p0.grad

            err = (g_adj - g_ref).abs().max() / g_ref.abs().max()
            assert err < 1e-12, (bc, n_T, float(err))


def test_gradcheck():
    n, k2, beta, kernel = _setup(n=8)
    idx = torch.tensor([9, 20, 35, 50])
    p0 = torch.randn(n, n, dtype=DTYPE, requires_grad=True)
    for bc in BC_TYPES:
        assert torch.autograd.gradcheck(
            lambda p: WaveSensorOp.apply(p, idx, 6, k2, beta, kernel, bc), (p0,), eps=1e-6, atol=1e-8
        ), bc


def test_sensor_operator():
    """G p0 reproduces the traces and G^T r reproduces the adjoint gradient."""
    n, k2, beta, kernel = _setup()
    idx = torch.randperm(n * n)[:5]
    for bc in BC_TYPES:
        for n_T in (1, 2, 3, 10):
            G = build_sensor_operator(idx, (n, n), n_T, k2, beta, kernel, bc_type=bc)
            p0 = torch.randn(n, n, dtype=DTYPE)
            traces = record_traces(p0, idx, n_T, k2, beta, kernel, bc_type=bc)
            assert torch.allclose((G @ p0.reshape(-1)).view(-1, n_T), traces, rtol=0, atol=1e-12), (bc, n_T)
            r = torch.randn(len(idx), n_T, dtype=DTYPE)
            g = adjoint_gradient(r, idx, (n, n), n_T, k2, beta, kernel, bc_type=bc)
            assert torch.allclose(G.T @ r.reshape(-1), g.reshape(-1), rtol=0, atol=1e-12), (bc, n_T)


def test_dpnn_gradient_modes_agree():
    """One training closure gives the same parameter gradients in all three modes."""
    from diffsfr.dpnn import DPNN
    from diffsfr.utils import Grid, make_tensor

    n_L, n_T = 20, 10
    x = np.linspace(0, 1, n_L)
    t = np.linspace(0, 0.5 * (n_T - 1) / (n_L - 1), n_T)
    xx, yy = np.meshgrid(x, x)
    grid = Grid(xx, yy, t, np.column_stack((xx.ravel(), yy.ravel())))
    grid.xy = make_tensor(grid.xy, "cpu")

    class Array:  # minimal stand-in for SensorArray
        idx = [45, 88, 123, 210, 301, 377]
        data = torch.randn(6, n_T)

    grads = {}
    for mode in ("autograd", "adjoint", "operator"):
        torch.manual_seed(1)
        model = DPNN(1.0, t[-1], 1.0, num_layers=2, num_hidden=16, device="cpu", gradient=mode)
        model.prepare(Array, grid)
        model.closure(Array, grid, update_rate=1)  # also exercises update_lambda
        grads[mode] = torch.cat([p.grad.reshape(-1) for p in model.nn.parameters()])
    scale = grads["autograd"].abs().max()
    for mode in ("adjoint", "operator"):
        err = (grads[mode] - grads["autograd"]).abs().max() / scale
        assert err < 1e-4, (mode, float(err))  # float32 network


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("passed:", name)
