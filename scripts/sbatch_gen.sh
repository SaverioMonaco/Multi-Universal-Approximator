#!/bin/bash
# Run one generative study on Maxwell (CPU), e.g.
#     sbatch scripts/sbatch_gen.sh --preset mnist_d32
#SBATCH --partition=allcpu
#SBATCH --job-name=mua_gen
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
# run from the repo root (sbatch uses the submission directory)
PYTHON=${MUA_PYTHON:-python}
echo "node $(hostname), cpus $(nproc), args $*"
$PYTHON -u scripts/gen.py "$@"
