#!/bin/bash
# All the studies of the thesis section. Fits and CLIC run locally, MNIST through Slurm.
set -e
cd "$(dirname "$0")/.."
PYTHON=${MUA_PYTHON:-python}
mkdir -p results/logs
$PYTHON scripts/fit.py
$PYTHON scripts/gen.py --preset clic_q8_d8
$PYTHON scripts/gen.py --preset clic_q25_d22
sbatch --job-name=mua_mnist_d16 scripts/sbatch_gen.sh --preset mnist_d16
sbatch --job-name=mua_mnist_d32 scripts/sbatch_gen.sh --preset mnist_d32
# when the jobs are done: $PYTHON scripts/plot.py   (figures in results/plots)
#                         $PYTHON scripts/export.py --dest DIR
