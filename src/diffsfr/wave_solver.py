"""Finite-difference wave solver with a hand-written discrete adjoint.

The 2D leapfrog scheme used by the DPNN is linear in the pressure and its
coefficients do not depend on the field:

    p^1 = C p^0,        p^n = A p^{n-1} + B p^{n-2},   n = 2, ..., N-1,

where A, B, C include the boundary condition. The gradient of a loss that only
sees the sensor traces M p^n therefore follows from the backward recursion

    mu^k = s^k + A^T mu^{k+1} + B^T mu^{k+2},   mu^N = mu^{N+1} = 0,
    dJ/dp^0 = s^0 + C^T mu^1 + B^T mu^2,

with s^k = M^T dJ/d(M p^k). The backward pass needs only the sensor residuals
and three field-sized buffers; no forward field is stored.

Contents
--------
time_integrator         full field, differentiable by autograd (reference path)
record_traces           sensor traces only, three rolling buffers, no graph
adjoint_gradient        backward recursion above
WaveSensorOp            torch.autograd.Function: forward = record_traces,
                        backward = adjoint_gradient
build_sensor_operator   the fixed linear map p^0 -> traces as a matrix G,
                        built with one batched adjoint run

All step functions accept fields of shape (..., H, W); leading dimensions are
treated as a batch.
"""

import torch
import torch.nn.functional as F

BC_TYPES = ("dirichlet", "neumann", "absorbing")

# Boundary index (b) and its inward neighbour (b_in) for each edge of a (..., H, W) field.
_COL_FIRST = ((Ellipsis, slice(None), 0), (Ellipsis, slice(None), 1))
_COL_LAST = ((Ellipsis, slice(None), -1), (Ellipsis, slice(None), -2))
_ROW_FIRST = ((Ellipsis, 0, slice(None)), (Ellipsis, 1, slice(None)))
_ROW_LAST = ((Ellipsis, -1, slice(None)), (Ellipsis, -2, slice(None)))


def laplacian_kernel(dtype, device):
    """3x3 five-point Laplacian stencil as a conv2d weight of shape (1, 1, 3, 3)."""
    return torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)


def solver_coefficients(c, dt, dx):
    """Return (k2, beta): squared Courant number and first-order absorbing-boundary coefficient."""
    k2 = float((c * dt / dx) ** 2)
    beta = float((c * dt - dx) / (c * dt + dx))
    return k2, beta


def _check_bc(bc_type):
    if bc_type not in BC_TYPES:
        raise ValueError("Unknown bc_type: %s. Supported: dirichlet, neumann, absorbing." % bc_type)


def _conv_valid(p, kernel):
    """Apply the stencil to (..., H, W) and keep the interior: (..., H-2, W-2)."""
    H, W = p.shape[-2:]
    out = F.conv2d(p.reshape(-1, 1, H, W), kernel)
    return out.reshape(*p.shape[:-2], H - 2, W - 2)


def _conv_valid_T(m, kernel):
    """Exact transpose of `_conv_valid`: (..., H-2, W-2) -> (..., H, W)."""
    h, w = m.shape[-2:]
    out = F.conv_transpose2d(m.reshape(-1, 1, h, w), kernel)
    return out.reshape(*m.shape[:-2], h + 2, w + 2)


# ----------------------------------------------------------------------------- forward step
def _step(p_n, p_nm1, k2, beta, kernel, bc_type, first):
    """One leapfrog step: returns p^{n+1} from p^n and p^{n-1}.

    Identical arithmetic to the original `_time_integrator` loop body, including the
    order in which the boundary edges are written. `p_nm1` is ignored when `first`.
    """
    lap = _conv_valid(p_n, kernel)
    if first:
        q_inner = p_n[..., 1:-1, 1:-1] + 0.5 * k2 * lap
    else:
        q_inner = 2 * p_n[..., 1:-1, 1:-1] - p_nm1[..., 1:-1, 1:-1] + k2 * lap

    q = p_n.clone()
    q[..., 1:-1, 1:-1] = q_inner

    if bc_type == "dirichlet":
        q[..., 0, :] = 0.0
        q[..., -1, :] = 0.0
        q[..., :, 0] = 0.0
        q[..., :, -1] = 0.0
    elif bc_type == "neumann":
        q[..., :, 0] = q[..., :, 1]
        q[..., :, -1] = q[..., :, -2]
        q[..., 0, :] = q[..., 1, :]
        q[..., -1, :] = q[..., -2, :]
    elif bc_type == "absorbing":
        # first-order absorbing boundary condition
        q[..., :, 0] = p_n[..., :, 1] + beta * (q[..., :, 1] - p_n[..., :, 0])
        q[..., :, -1] = p_n[..., :, -2] + beta * (q[..., :, -2] - p_n[..., :, -1])
        q[..., 0, :] = p_n[..., 1, :] + beta * (q[..., 1, :] - p_n[..., 0, :])
        q[..., -1, :] = p_n[..., -2, :] + beta * (q[..., -2, :] - p_n[..., -1, :])
    else:
        _check_bc(bc_type)
    return q


# ----------------------------------------------------------------------------- transposed step
def _edge_ops(bc_type, beta):
    """Edge assignments in the order `_step` writes them.

    Each entry (b, b_in, a1, a2, a3) encodes  q[b] = a1 * p_n[b_in] + a2 * p_n[b] + a3 * q[b_in].
    """
    if bc_type == "dirichlet":
        return [(*e, 0.0, 0.0, 0.0) for e in (_ROW_FIRST, _ROW_LAST, _COL_FIRST, _COL_LAST)]
    if bc_type == "neumann":
        return [(*e, 0.0, 0.0, 1.0) for e in (_COL_FIRST, _COL_LAST, _ROW_FIRST, _ROW_LAST)]
    if bc_type == "absorbing":
        return [(*e, 1.0, -beta, beta) for e in (_COL_FIRST, _COL_LAST, _ROW_FIRST, _ROW_LAST)]
    _check_bc(bc_type)


def _step_T(mu, k2, beta, kernel, bc_type, first):
    """Exact transpose of `_step`.

    Given the adjoint `mu` of p^{n+1}, returns (A^T mu, B^T mu), the contributions to the
    adjoints of p^n and p^{n-1}. For the first step returns (C^T mu, None).

    The step is a composition of linear stages (copy, interior update, four edge
    overwrites); the transpose applies their transposes in reverse order. Undoing an
    overwrite q[b] <- f(...) reads the adjoint of q[b], zeroes it (the old value was
    discarded) and distributes it to everything f depended on.
    """
    mu = mu.clone()
    nu = torch.zeros_like(mu)

    for b, b_in, a1, a2, a3 in reversed(_edge_ops(bc_type, beta)):
        w = mu[b].clone()
        mu[b] = 0.0
        if a1 != 0.0:
            nu[b_in] += a1 * w
        if a2 != 0.0:
            nu[b] += a2 * w
        if a3 != 0.0:
            mu[b_in] += a3 * w

    m = mu[..., 1:-1, 1:-1].clone()
    mu[..., 1:-1, 1:-1] = 0.0
    nu += mu  # the initial copy q = p_n.clone(): what remains of mu lives on the boundary
    lap_T = _conv_valid_T(m, kernel)

    if first:
        nu[..., 1:-1, 1:-1] += m
        nu += (0.5 * k2) * lap_T
        return nu, None

    nu[..., 1:-1, 1:-1] += 2 * m
    nu += k2 * lap_T
    nu_prev = torch.zeros_like(nu)
    nu_prev[..., 1:-1, 1:-1] = -m
    return nu, nu_prev


# ----------------------------------------------------------------------------- solvers
def time_integrator(p0, n_T, cdtdx2, kernel, beta, bc_type="absorbing"):
    """FDTD solver for the 2D wave equation; returns all frames stacked on the last axis.

    For p0 of shape (H, W) the result has shape (H, W, n_T), as in the original
    implementation. Differentiable by autograd (stores every frame).
    """
    _check_bc(bc_type)
    frames = [p0]
    for t in range(1, n_T):
        p_nm1 = frames[-2] if t > 1 else None
        frames.append(_step(frames[-1], p_nm1, cdtdx2, beta, kernel, bc_type, first=(t == 1)))
    return torch.stack(frames, dim=-1)


def record_traces(p0, idx, n_T, k2, beta, kernel, bc_type="absorbing"):
    """Run the solver keeping two frames in memory; return sensor traces (M_ob, n_T).

    `idx` indexes the flattened (row-major) field, as `SensorArray.idx` does.
    """
    _check_bc(bc_type)
    traces = p0.new_empty(idx.numel(), n_T)
    traces[:, 0] = p0.reshape(-1)[idx]
    p_nm1, p_n = None, p0
    for t in range(1, n_T):
        p_np1 = _step(p_n, p_nm1, k2, beta, kernel, bc_type, first=(t == 1))
        traces[:, t] = p_np1.reshape(-1)[idx]
        p_nm1, p_n = p_n, p_np1
    return traces


def adjoint_gradient(grad_traces, idx, shape, n_T, k2, beta, kernel, bc_type="absorbing"):
    """Gradient with respect to p^0 of a loss whose gradient w.r.t. the traces is `grad_traces`.

    Implements the backward recursion in "push" form with three field-sized buffers:
    when the loop runs for index t, `acc_cur` equals mu^t and `acc_prev` holds
    s^{t-1} + B^T mu^{t+1}.
    """
    _check_bc(bc_type)
    H, W = shape

    def source(t):  # s^t = M^T grad_traces[:, t]
        s = grad_traces.new_zeros(H * W)
        s.index_add_(0, idx, grad_traces[:, t])
        return s.view(H, W)

    if n_T == 1:
        return source(0)
    if n_T == 2:
        return source(0) + _step_T(source(1), k2, beta, kernel, bc_type, first=True)[0]

    acc_cur, acc_prev = source(n_T - 1), source(n_T - 2)
    for t in range(n_T - 1, 1, -1):
        a_T, b_T = _step_T(acc_cur, k2, beta, kernel, bc_type, first=False)
        acc_cur, acc_prev = acc_prev + a_T, source(t - 2) + b_T
    c_T, _ = _step_T(acc_cur, k2, beta, kernel, bc_type, first=True)
    return acc_prev + c_T


class WaveSensorOp(torch.autograd.Function):
    """Sensor traces of the wave solver, with the discrete adjoint as backward pass.

    traces = WaveSensorOp.apply(p0, idx, n_T, k2, beta, kernel, bc_type)

    p0: (H, W) initial field; idx: LongTensor of flattened sensor indices.
    Returns (M_ob, n_T). Memory: three fields in each pass, independent of n_T.
    """

    @staticmethod
    def forward(ctx, p0, idx, n_T, k2, beta, kernel, bc_type):
        if p0.dim() != 2:
            raise ValueError("WaveSensorOp expects a 2D field of shape (H, W).")
        with torch.no_grad():
            traces = record_traces(p0.detach(), idx, n_T, k2, beta, kernel, bc_type)
        ctx.save_for_backward(idx, kernel)
        ctx.n_T, ctx.k2, ctx.beta, ctx.bc_type = n_T, k2, beta, bc_type
        ctx.shape = tuple(p0.shape)
        return traces

    @staticmethod
    def backward(ctx, grad_traces):
        idx, kernel = ctx.saved_tensors
        grad_p0 = adjoint_gradient(
            grad_traces.contiguous(), idx, ctx.shape, ctx.n_T, ctx.k2, ctx.beta, kernel, ctx.bc_type
        )
        return grad_p0, None, None, None, None, None, None


@torch.no_grad()
def build_sensor_operator(idx, shape, n_T, k2, beta, kernel, bc_type="absorbing"):
    """Matrix G of shape (M_ob * n_T, H * W) with traces.reshape(-1) = G @ p0.reshape(-1).

    Built with one batched adjoint run (batch = M_ob): a unit impulse at each sensor,
    nu_0 = e_m, is propagated by nu_1 = A^T nu_0, nu_k = A^T nu_{k-1} + B^T nu_{k-2}, and

        G[m, 0] = e_m,  G[m, 1] = C^T nu_0,  G[m, n] = C^T nu_{n-1} + B^T nu_{n-2}  (n >= 2).

    Valid because the solver is linear with fixed coefficients. Memory M_ob * n_T * H * W.
    """
    _check_bc(bc_type)
    H, W = shape
    n_ob = idx.numel()
    G = kernel.new_zeros(n_ob, n_T, H * W)

    nu0 = kernel.new_zeros(n_ob, H * W)
    nu0[torch.arange(n_ob, device=idx.device), idx] = 1.0
    G[:, 0] = nu0
    nu0 = nu0.view(n_ob, H, W)
    if n_T == 1:
        return G.view(n_ob * n_T, H * W)

    G[:, 1] = _step_T(nu0, k2, beta, kernel, bc_type, first=True)[0].reshape(n_ob, -1)
    a_prev, b_prev = _step_T(nu0, k2, beta, kernel, bc_type, first=False)  # A^T nu_0, B^T nu_0
    b_prev2 = None
    for n in range(2, n_T):
        nu = a_prev if b_prev2 is None else a_prev + b_prev2  # nu_{n-1}
        c_T, _ = _step_T(nu, k2, beta, kernel, bc_type, first=True)
        G[:, n] = (c_T + b_prev).reshape(n_ob, -1)
        b_prev2 = b_prev
        a_prev, b_prev = _step_T(nu, k2, beta, kernel, bc_type, first=False)
    return G.view(n_ob * n_T, H * W)
