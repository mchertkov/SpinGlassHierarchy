#!/bin/bash
# Campaign N4 (2026-11-01): SK glasses at N=16 (structure on the 9-point grid) and N=20 (structure, dynamics, per-trajectory counts),
# to compare partial commitments in sparse (Bethe) and dense (SK) glasses. About 45 min on two cores.
cd "$(dirname "$0")/.."; mkdir -p logs
P=experiments/n1_levels.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
B9="0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0"
( python3 $P structure --family sk --N 16 --betas $B9 --seeds 1-30
  python3 $P structure --family sk --N 20 --betas $B9 --seeds 1-24
  python3 experiments/n2_paths.py --family sk --N 20 --betas 1.0 1.5 2.0 3.0 --seeds 1-24 --paths 64 ) > logs/n4_a.log 2>&1 &
( python3 $P scan --family sk --N 20 --betas 0.5 1.0 1.5 2.0 3.0 --seeds 1-24 --paths 96 ) > logs/n4_b.log 2>&1 &
wait
