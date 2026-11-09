#!/bin/bash
# Campaign N1 (exact level-resolved decisions); total about 2.5 h on two cores.
cd "$(dirname "$0")/.."
P=experiments/n1_levels.py
python3 $P scan --family bethe  --N 16 --betas 0.5 1.0 1.5 2.0 2.5 --seeds 1-60 --paths 256
python3 $P scan --family ferro  --N 16 --betas 0.5 1.0 1.5 2.0 2.5 --seeds 1-30 --paths 256
python3 $P scan --family mattis --N 16 --betas 1.37 --seeds 1-10 --paths 256
python3 $P anatomy --family bethe --N 24 --beta 2.5 --seed 2 --paths 128 --cov 16
python3 $P anatomy --family bethe --N 16 --beta 1.5 --seed 49 --paths 512 --cov 32
python3 $P scan --family bethe  --N 20 --betas 0.5 1.0 1.5 2.0 2.5 --seeds 1-24 --paths 96
python3 $P scan --family sk     --N 16 --betas 0.5 1.0 1.5 2.0 3.0 --seeds 1-30 --paths 256
python3 $P scan --family bethe  --N 24 --betas 1.0 2.5 --seeds 1-6 --paths 48
python3 $P scan --family ferro  --N 20 --betas 1.0 2.5 --seeds 1-12 --paths 96
