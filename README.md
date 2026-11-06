# Multi-Universal Approximator
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16027822.svg)](https://doi.org/10.5281/zenodo.16027822)

This repository implements a Multi-Universal Approximator using parameterized quantum circuits. The project explores classically-simulable, entanglement-free, quantum circuits as universal generative models for continuous multivariate distributions.

## Installation

#### with uv
```bash
uv sync
uv pip install -e .
```

#### pip
Create an environment and run
```bash
pip install -e .
```
## Examples

The circuit is simulated exactly as a 2-component state vector in JAX
(`mua.unitaries.Uuat.Uuat`); the PennyLane QNode `Uuat_qml` gives the same result
(`tests/test_backend.py`) and is only used for drawings.

### Fitting a 1D Function
You can approximate any function $f: [-1, 1] \to [-1, 1]$.

```python
from mua.train import regression
from mua.plots.fit import show
import numpy as np

def tanh(x):
    return np.tanh(5 * x)

# depth 5 (layers of the UAT ansatz), 150 epochs, learning rate 0.1
params, history = regression.train(tanh, depth=5, num_epochs=150, lr=0.1, seed=42)
# history["f"][k], history["phi"][k], ...: prediction and parameters after k epochs

show(tanh, params)  # predicted and target lines
```

![Approximation plot](notebooks/examples/fits/fit1.png)

### Generation
```python
from mua.train import gen
import numpy as np

data = ...  # (num_samples, num_qubits) training vectors in [-1, 1], see mua.scalers

params, history = gen.train(
    data,
    depth=8,          # layers per qubit = dimension of the shared noise
    num_epochs=50,    # passes over the data, one Adam step per batch
    batch_size=2000,  # samples per MMD estimate
    lr=5e-3,
    w_scale=1.0,      # w ~ w_scale U[0,1); use ~1/depth for deep circuits
    test_data=...,    # held-out vectors for the per-epoch MMD / correlation distance
    seed=42,
)
fake = gen.generate(params, 1000, np.random.default_rng(0))
```

![Generation plot](notebooks/examples/generations/fake_images.png)

### Studies (`scripts/`)
The studies of the thesis are scripts writing JSON to `results/` (gitignored):

```bash
python scripts/fit.py                         # 4 regressions -> results/fits/
python scripts/gen.py --preset clic_q8_d8     # CLIC 8 px     -> results/clic/
python scripts/gen.py --preset clic_q25_d22   # CLIC 25 px
sbatch scripts/sbatch_gen.sh --preset mnist_d16   # full MNIST (784 qubits), Slurm
sbatch scripts/sbatch_gen.sh --preset mnist_d32
python scripts/export.py --dest DIR           # copy the JSON elsewhere (or MUA_EXPORT_DIR)
```
`scripts/run_all.sh` runs everything. Data paths: `MUA_CLIC_DIR` (folder with
`clic_<n>px_{train,test}.h5`, from [desyqml/clic](https://doi.org/10.5281/zenodo.16027525)) and
`MUA_MNIST` (`mnist.npz`), see `mua.data`.

## Sources

It is based on the following research papers:

* Parameterized quantum circuits as universal generative models for continuous multivariate distributions. [Read the paper](https://arxiv.org/pdf/2402.09848)
* One qubit as a Universal Approximant. [Read the paper](https://arxiv.org/pdf/2102.04032)

# Citation
If you use this software in your research or publications, **please cite** the following: 

```
@software{monaco_2025_16027822,
  author       = {Monaco, Saverio and
                  Slim, Jamal and
                  Krücker, Dirk and
                  Borras, Kerstin},
  title        = {Multi-Universal-Approximator},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.16027822},
  url          = {https://doi.org/10.5281/zenodo.16027822},
  swhid        = {swh:1:dir:f94c2fad8e2f8b4fbf98a993a504c147e4d00684
                   ;origin=https://doi.org/10.5281/zenodo.16027821;vi
                   sit=swh:1:snp:ffcaaeef88dbf3c5b7b155807d8d5a43f743
                   7e6b;anchor=swh:1:rel:876569afb971342040497924feca
                   8b9ba6491835;path=desyqml-Multi-Universal-
                   Approximator-52207f4
                  },
}
```
