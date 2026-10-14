import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, animation

def _save_show(figname, show):
    """Save the current figure to path specified by figname and/or show it."""
    if figname:
        plt.savefig(figname, format="pdf", bbox_inches="tight")
    if show:
        plt.show()


def plot_field(p, L, T, figname=None, show=False):
    """Plot four snapshots of the field p (3 evenly spaced + final)."""
    n_t = p.shape[-1]
    time_axis = np.linspace(0, T, n_t)
    step = int(np.floor(n_t / 3))

    p_max = 1  # fixed scale (change if needed)

    fig, axs = plt.subplots(1, 4, figsize=(8, 3))
    fig.tight_layout()

    for idx in range(3):
        t_index = idx * step
        axs[idx].imshow(p[:, :, t_index], vmin=-p_max, vmax=p_max, origin="lower", cmap="seismic", extent=[0, L, 0, L])
        axs[idx].set_title(f"t: {time_axis[t_index]:.3f}")

    axs[3].imshow(p[:, :, -1], vmin=-p_max, vmax=p_max, origin="lower", cmap="seismic", extent=[0, L, 0, L])
    axs[3].set_title(f"t: {time_axis[-1]:.3f}")

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.97, 0.31, 0.02, 0.45])
    fig.colorbar(axs[3].images[0], cax=cbar_ax)

    _save_show(figname, show)


def plot_data(p_data, xy_data, L, T, figname=None, show=False):
    """Plot sensor positions and observed time-series data."""
    time_axis = np.linspace(0, T, p_data.shape[-1])

    fig, (ax_map, ax_timeseries) = plt.subplots(1, 2, figsize=(6, 3))
    fig.tight_layout()

    ax_map.set_title("(a) Array")
    ax_map.set_ylabel("y")
    ax_map.set_xlabel("x")
    ax_map.scatter(xy_data[:, 0], xy_data[:, 1], s=10, c="black", marker="o")
    ax_map.set_xlim([0, L])
    ax_map.set_ylim([0, L])

    ax_timeseries.set_title("(b) Observed data")
    ax_timeseries.set_xlabel("Time")
    ax_timeseries.plot(time_axis, p_data.T)
    ax_timeseries.set_xlim([0, T])

    _save_show(figname, show)


def plot_train_log(pinn, dpnn, label_pinn, label_dpnn, figname=None, show=False):
    """Plot training histories for PINN and DPNN (losses and lambdas)."""
    fig, axs = plt.subplots(3, 2, figsize=(6, 8))

    skip = 500

    # DPNN weighted losses
    dpnn_loss = dpnn.loss_hist
    dpnn_lamb = dpnn.lamb_hist
    color_cycle = ["tab:blue", "tab:green"]
    for i in range(len(dpnn_loss)):
        loss_arr = np.asarray(dpnn_loss[i])
        lamb_arr = np.asarray(dpnn_lamb[i])
        weighted = loss_arr * lamb_arr / np.asarray(dpnn_lamb[0])
        axs[0, 0].plot(np.arange(len(loss_arr))[::skip], weighted[::skip], label=label_dpnn[i], alpha=1, color=color_cycle[i])
    axs[0, 0].legend()
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("Optimizer steps")
    axs[0, 0].set_ylabel("Weighted loss")
    axs[0, 0].set_title("(a) DPNN loss")

    # DPNN lambdas
    for i in range(len(dpnn_lamb)):
        axs[0, 1].plot(np.arange(len(dpnn_lamb[i]))[::skip], np.asarray(dpnn_lamb[i])[::skip], alpha=1, color=color_cycle[i])
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlabel("Optimizer steps")
    axs[0, 1].set_ylabel("Weighting params. $\\lambda$")
    axs[0, 1].set_title("(b) DPNN weighting $\\lambda$")

    # PINN weighted losses
    pinn_loss = pinn.loss_hist
    pinn_lamb = pinn.lamb_hist
    for i in range(len(pinn_loss)):
        loss_arr = np.asarray(pinn_loss[i])
        lamb_arr = np.asarray(pinn_lamb[i])
        weighted = loss_arr * lamb_arr / np.asarray(pinn_lamb[0])
        axs[1, 0].plot(np.arange(len(loss_arr))[::skip], weighted[::skip], label=label_pinn[i], alpha=0.7)
    axs[1, 0].legend()
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("Optimizer steps")
    axs[1, 0].set_ylabel("Weighted loss")
    axs[1, 0].set_title("(c) PINN loss")

    # PINN lambdas
    for i in range(len(pinn_lamb)):
        axs[1, 1].plot(np.arange(len(pinn_lamb[i]))[::skip], np.asarray(pinn_lamb[i])[::skip], alpha=1)
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("Optimizer steps")
    axs[1, 1].set_ylabel("Weighting params. $\\lambda$")
    axs[1, 1].set_title("(d) PINN weighting $\\lambda$")

    # Total loss comparison
    dpnn_total = np.zeros_like(np.asarray(dpnn_loss[0]))
    for i in range(len(dpnn_loss)):
        dpnn_total += np.asarray(dpnn_loss[i]) * np.asarray(dpnn_lamb[i]) / np.asarray(dpnn_lamb[0])
    axs[2, 0].plot(np.arange(len(dpnn_total))[::skip], dpnn_total[::skip], label="dpnn", alpha=1, color="tab:cyan")

    pinn_total = np.zeros_like(np.asarray(pinn_loss[0]))
    for i in range(len(pinn_loss)):
        pinn_total += np.asarray(pinn_loss[i]) * np.asarray(pinn_lamb[i]) / np.asarray(pinn_lamb[0])
    axs[2, 0].plot(np.arange(len(pinn_total))[::skip], pinn_total[::skip], label="pinn", alpha=1, color="tab:pink")

    axs[2, 0].legend()
    axs[2, 0].set_yscale("log")
    axs[2, 0].set_xlabel("Optimizer steps")
    axs[2, 0].set_ylabel("Total loss")
    axs[2, 0].set_title("(e) Total loss")

    for ax in axs.flat:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    axs[2, 1].axis("off")

    plt.tight_layout()
    _save_show(figname, show)


def plot_compare_field(p_true, p_pinn, p_dpnn, t, n_T, L, n_L, figname=None, show=False):
    """Plot panel comparing true / DPNN / PINN fields and errors."""
    p_true = p_true.reshape(n_L, n_L, n_T)
    p_pinn = p_pinn.reshape(n_L, n_L, n_T)
    p_dpnn = p_dpnn.reshape(n_L, n_L, n_T)

    extent = (0, L, 0, L)
    cmap = "seismic"
    cmap_error = "PRGn"
    p_max = 1
    norm = colors.Normalize(vmin=-p_max, vmax=p_max)
    error_max = 0.5
    norm_error = colors.TwoSlopeNorm(vmin=-error_max, vcenter=0.0, vmax=error_max)
    step = int(np.floor(n_T / 3))

    fig, axs = plt.subplots(5, 4, figsize=(6, 9), constrained_layout=True)
    for i in range(3):
        t_index = i * step
        im = axs[0, i].imshow(p_true[:, :, t_index], extent=extent, cmap=cmap, norm=norm, origin="lower")
        axs[1, i].imshow(p_dpnn[:, :, t_index], extent=extent, cmap=cmap, norm=norm, origin="lower")
        axs[2, i].imshow(p_pinn[:, :, t_index], extent=extent, cmap=cmap, norm=norm, origin="lower")

        im2 = axs[3, i].imshow(p_true[:, :, t_index] - p_dpnn[:, :, t_index], extent=extent, cmap=cmap_error, norm=norm_error, origin="lower")
        axs[4, i].imshow(p_true[:, :, t_index] - p_pinn[:, :, t_index], extent=extent, cmap=cmap_error, norm=norm_error, origin="lower")

        if i == 0:
            axs[0, i].set_title(f"(a) p_true t: {t[t_index]:.3f}", loc="left")
            axs[1, i].set_title("(b) p_dp", loc="left")
            axs[2, i].set_title("(c) p_pinn", loc="left")
            axs[3, i].set_title("(d) error_dp", loc="left")
            axs[4, i].set_title("(e) error_pinn", loc="left")
        else:
            axs[0, i].set_title(f"t: {t[t_index]:.3f}")

    im = axs[0, 3].imshow(p_true[:, :, -1], extent=extent, cmap=cmap, norm=norm, origin="lower")
    axs[1, 3].imshow(p_dpnn[:, :, -1], extent=extent, cmap=cmap, norm=norm, origin="lower")
    axs[2, 3].imshow(p_pinn[:, :, -1], extent=extent, cmap=cmap, norm=norm, origin="lower")
    im2 = axs[3, 3].imshow(p_true[:, :, -1] - p_dpnn[:, :, -1], extent=extent, cmap=cmap_error, norm=norm_error, origin="lower")
    axs[4, 3].imshow(p_true[:, :, -1] - p_pinn[:, :, -1], extent=extent, cmap=cmap_error, norm=norm_error, origin="lower")
    axs[0, 3].set_title(f"t: {t[-1]:.3f}")

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    cbar_ax = fig.add_axes([0.9999, 0.815, 0.02, 0.158])
    cbar_ax2 = fig.add_axes([0.9999, 0.215, 0.02, 0.158])
    fig.colorbar(im, cax=cbar_ax)
    cbar = fig.colorbar(im2, cax=cbar_ax2)
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])

    _save_show(figname, show)

def animate_comparison(p_true, p_pinn, p_dpnn, L, figname):
    """Animation comparing true field, PINN and DPNN."""

    frames = []
    fig, axs = plt.subplots(2, 3, gridspec_kw={"width_ratios":[0.33,0.33,0.33]}, figsize=(8,6))
    axs[0, 0].set_title('reference')
    axs[0, 1].set_title('pinn')
    axs[0, 2].set_title('dpnn')
    axs[1, 1].set_title('error pinn')
    axs[1, 2].set_title('error dpnn')
    axs[1, 0].axis("off")

    norm = colors.Normalize(vmin=-1, vmax=1)
    error_max = 0.5
    norm_error = colors.TwoSlopeNorm(vmin=-error_max, vcenter=0.0, vmax=error_max)

    for i in range(p_true.shape[2]):
        im1 = axs[0, 0].imshow(p_true[:,:,i], norm=norm, origin='lower', cmap='seismic', extent=[0, L, 0, L], animated=True)
        im2 = axs[0, 1].imshow(p_pinn[:,:,i], norm=norm, origin='lower', cmap='seismic', extent=[0, L, 0, L], animated=True)
        im3 = axs[0, 2].imshow(p_dpnn[:,:,i], norm=norm, origin='lower', cmap='seismic', extent=[0, L, 0, L], animated=True)
        im4 = axs[1, 1].imshow(p_true[:,:,i]- p_pinn[:,:,i], norm=norm_error, origin='lower', cmap='PRGn', extent=[0, L, 0, L], animated=True)
        im5 = axs[1, 2].imshow(p_true[:,:,i]- p_dpnn[:,:,i], norm=norm_error, origin='lower', cmap='PRGn', extent=[0, L, 0, L], animated=True)
        frames.append([im1, im2, im3, im4, im5])

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    writer = animation.PillowWriter(fps=60)
    ani.save(figname, writer=writer)
    plt.close()

