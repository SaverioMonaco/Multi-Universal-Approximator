#!/bin/bash
# Run one generative study on a Maxwell GPU node, e.g.
#     MUA_PYTHON=/path/to/env-with-jax[cuda12]/bin/python sbatch scripts/sbatch_gen_gpu.sh --preset mnist_d32
#SBATCH --partition=allgpu
#SBATCH --job-name=mua_gen_gpu
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
# run from the repo root (sbatch uses the submission directory)
PYTHON=${MUA_PYTHON:-python}
echo "node $(hostname), args $*"
nvidia-smi -L
$PYTHON -c "import jax; print('jax devices:', jax.devices())"
$PYTHON -u scripts/gen.py "$@"
