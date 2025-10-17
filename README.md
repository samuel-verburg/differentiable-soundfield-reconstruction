## Differentiable Physics for Sound Field Reconstruction

This repository allows you to solve sound field reconstruction problems using **differentiable-physics to train neural networks** (DPNN).

The DPNN approach is described in detail here:

S. A. Verburg, E. Fernandez-Grande, P. Gerstoft, "Differentiable physics for sound field reconstruction", J. Acoust. Soc. Am. XX, XX (2025).

<p align="center">
    <img src="../examples/figures/twenty_pulses/animation.gif" alt="Sound field reconstructed by PINN and DPNN models" width=50%>
</p>

This repository implements two deep-learning approaches for reconstructing acoustic fields from sparse sensor measurements:

- PINN — physics-informed neural network that models $p(x,y,t)$ (`src/pinn.py`).
- DPNN — physics-constrained neural network that models the initial pressure $p(x,y,0)$ and uses a differentiable FDTD solver to impose physics (`src/dpnn.py`).

### 📁 Repo layout

- `src/` — core implementation
	- `pinn.py` — PINN implementation, training loop, save/load helpers
	- `dpnn.py` — DPNN implementation and explicit time-integrator
	- `fields.py` — `SoundField` container and generation 
	- `sensors.py` — `SensorArray` sensor distributions and observed data 
	- `utils.py` — helpers
	- `plots.py` — plotting

- `examples/` — `example.ipynb` demonstrates the workflow: build/load field → sensor sampling → train PINN/DPNN → compare.
- `data/` — pickled `SoundField` instances used by examples.
- `models/` — example trained models and training histories (loss and lambda arrays).


### ⚙️ Installation

1. Clone the repository (SSH or HTTPS):

```bash
# SSH (if you have SSH keys configured)
git clone git@github.com:samuel-verburg/differentiable-soundfield-reconstruction.git

# OR HTTPS
git clone https://github.com/samuel-verburg/differentiable-soundfield-reconstruction.git
```
2. Create and activate a virtual environment 

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. (Recommended) Install a CUDA-aware PyTorch wheel first if you plan to use a GPU.
Visit https://pytorch.org/get-started/locally/ and follow the selector to get the right pip command for your CUDA/toolkit. 

For CPU-only:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

4. Install the package in editable mode from the repository root 
```bash
# from repo root
pip install -e .
```

### 🚀 Quick start — run the example notebook

Start Jupyter in the repo root and open the example notebook:

```bash
jupyter notebook examples/example.ipynb
```
In the notebook, run cells in order. Notes:
- The notebook adds the repository root to `sys.path` so `import src.*` works when started from `examples/`.
- For quicker runs or debugging set `device='cpu'` in the notebook; the code defaults to GPU when available.

### ⚠️ Important points

- Do not change `Grid` attribute names (`xx, yy, t, xy, xyt, n_L, n_T, dx, dy, dt`) — many functions rely on them.
- The DPNN initializer enforces a CFL-like stability check on `grid.dt`; if you hit the stability error, adjust `dt` or increase spatial resolution (smaller `dx`).
- Model saving: `PINN.save_nn(path)` and `DPNN.save_nn(path)` write a `.pt` file and two history files named `*_loss_hist.npy` and `*_lamb_hist.npy` — preserve these names if you change save/load logic.

### 👥 Maintainers

Author / maintainer: Samuel A. Verburg — saveri@dtu.dk


### 📜 License

See the `LICENSE` file in the repository root.

### 📚 Citation

If you find our code useful and use them in your own work, please use the following citation entry:

```
@article{Verburg2025,
author = {Verburg, S. A. and Fernandez-Grande, Efren and Gerstoft, Peter},
doi = {},
journal = {J. Acoust. Soc. Am.},
number = {},
pages = {},
title = {Differentiable physics for sound field reconstruction},
url = {},
volume = {},
year = {2025}
}
```

---