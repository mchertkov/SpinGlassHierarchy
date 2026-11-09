#!/bin/bash
# Second-round extension, run as two single-threaded queues in parallel (structure | dynamics).
cd "$(dirname "$0")/.."; mkdir -p logs
P=experiments/n1_levels.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
B9="0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0"
( python3 $P structure --family bethe --N 16 --betas $B9 --seeds 1-60
  python3 $P structure --family bethe --N 20 --betas $B9 --seeds 1-24
  python3 $P structure --family ferro --N 16 --betas $B9 --seeds 1-30
  python3 $P structure --family bethe --N 24 --betas 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0 --seeds 2
  python3 $P structure --family bethe --N 24 --betas 1.0 2.5 1.5 2.0 --seeds 1-12
  python3 $P structure --family bethe --N 24 --betas 0.5 --seeds 1-12 ) > logs/n1ext_struct.log 2>&1 &
( python3 $P scan --family bethe --N 24 --betas 1.5 2.0 --seeds 1-6 --paths 48 ) > logs/n1ext_dyn.log 2>&1 &
wait
