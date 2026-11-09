#!/bin/bash
# Campaign N2 (2026-10-31): discretization (n2_discretization.py, run separately) and per-path decision counting.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python3 experiments/n2_paths.py --N 16 --betas 1.0 1.5 2.0 2.5 --seeds 1-60 --paths 128
python3 experiments/n2_paths.py --family ferro --N 16 --betas 1.0 2.5 --seeds 1-20 --paths 128
python3 experiments/n2_paths.py --N 20 --betas 1.5 2.5 --seeds 1-12 --paths 64
