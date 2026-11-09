#!/bin/bash
# Full recomputation of the archived data in results/, in execution order. NOT needed to redraw the figures
# (use scripts/make_figures.py). Times are wall-clock on two cores (2026, laptop-class CPU). Run from the repository root:
#   bash scripts/run_all_experiments.sh            # everything (exact part ~7 h; sampled and learned parts add several hours)
# Each block can also be run on its own; scripts skip results that already exist where noted.
set -e
cd "$(dirname "$0")/.."; mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# ---- 1. Exact campaign, N <= 24 (Figs. 2-8) ---------------------------------------------------------------------
bash experiments/n1_run_all.sh            # classes, decision times, trajectories: Bethe/ferro/Mattis/SK N=16, Bethe N=20,24  (~2.5 h) -> results/N1
bash experiments/n1_run_extension.sh      # classes on the nine-point temperature grid; more N=24 instances               (~1.5 h) -> results/N1S, N1
bash experiments/n2_run.sh                # per-trajectory commitments, Bethe and ferro N=16, Bethe N=20                    (~40 min) -> results/N2/paths_*
python3 experiments/n2_paths.py --family sk --N 16 --betas 1.0 1.5 2.0 3.0 --seeds 1-30 --paths 128                   # (~8 min)
python3 experiments/n2_discretization.py --betas 2.5 --seeds-per-beta 6 --M 4000                                     # (~40 min) -> results/N2/disc_*
python3 experiments/n2_discretization.py --betas 1.5 --seeds-per-beta 3 --M 2000                                     # (~30 min)
python3 experiments/n3_partial.py         # class posteriors of the N=24 instance for Fig. 3a-c                             (seconds) -> results/N3
bash experiments/n4_run_sk.sh             # SK N=16,20: classes, decision times, commitments                                (~1 h) -> results/N1, N1S, N2
python3 experiments/n4_families.py        # Bethe vs SK summary and Fig. 7                                                  (seconds) -> results/N4

# ---- 2. Positive control with a known tree (Appendix B) ----------------------------------------------------------
python3 experiments/e0_caricature.py      # (~3 min) -> results/E0
python3 experiments/s8_validation.py      # blind identification, structured and unselected controls (~12 min) -> results/S8

# ---- 3. Exact N=16 cohort for the static law and the contrast identities (Appendices C, D) -----------------------
python3 experiments/e5_exact_transients.py                    # (minutes) -> results/E5F
python3 experiments/s7_exact16_static.py; python3 experiments/s7_exact16_stats.py   # (minutes) -> results/S7

# ---- 4. Sampled posteriors at N = 32, 64 (Figs. 9, 10) ----------------------------------------------------------
python3 experiments/e5b_sequential_smc.py                     # equilibrium libraries and SMC posteriors (hours) -> results/E5B
python3 experiments/e5c_fieldwise_audit.py                    # fixed-field audits (about an hour) -> results/E5C
python3 experiments/e5d_analyze.py                            # information curves -> results/E5D
python3 experiments/e5e_matched_nulls.py; python3 experiments/e5e_random_bethe_audit.py   # controls (hours) -> results/E5E
python3 experiments/s7_e9_larger.py                           # label-free law on new instances (hours) -> results/S7

# ---- 5. Contrast identities and learned denoisers (Fig. 11, Appendix C) --------------------------------------------
python3 experiments/e6_pt_library.py                          # parallel-tempering libraries (hours) -> results/E6/libraries
python3 experiments/e6a_contrast_algorithms.py; python3 experiments/e6a_large_prediction.py; python3 experiments/e6a_calibration.py
python3 experiments/e6_reference_curves.py
python3 experiments/e6b_train.py; python3 experiments/e6b_analyze.py   # needs PyTorch (hours on CPU) -> results/E6/E6B
python3 experiments/s7_e10_learned.py                         # memorization study (needs PyTorch) -> results/S7

# ---- 6. Figures ------------------------------------------------------------------------------------------------
python3 scripts/make_figures.py
