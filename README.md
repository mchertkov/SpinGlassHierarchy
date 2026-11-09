# How Path Integral Diffusion Decides a Finite Spin Glass

Code, archived data, figure scripts and notebooks for the paper

> M. Chertkov, *How Path Integral Diffusion Decides a Finite Spin Glass: Stochastic Localization, Level-Resolved Information and
> Spectral Decision Times* (2026). arXiv: to be added.

## What the paper does

A diffusion-type generator that samples a frustrated spin glass must decide, at some point, which macroscopic alternatives its sample
will belong to. The paper studies which alternatives are decided first, and when, for Path Integral Diffusion (PID) from a point source.

* **Theory.** Every isotropic PID is a time-changed Gaussian stochastic localization: on the internal clock γ (the signal-to-noise
  ratio) all protocols are the same process, and every schedule γ(t) is realized by some PID protocol. The generator's state is described
  by its *partial commitments*, the probabilities it assigns to nested classes of configurations fixed by the target. They are
  martingales that move along class contrasts; their average gives the information about each level at an exact rate; and each split
  is decided, approximately, like a binary Gaussian channel at γ ≈ 1.044/λ_P, with λ_P the leading eigenvalue of the parent class's correlation matrix.
* **Experiments.** Exactly enumerated Bethe-lattice and Sherrington–Kirkpatrick glasses with up to 24 spins, sampled posteriors at
  N = 32, 64, matched controls (ferromagnet on the same graphs, Mattis model, independent spins) and small trained denoisers.

## Quick start

```bash
git clone https://github.com/mchertkov/SpinGlassHierarchy.git && cd SpinGlassHierarchy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # PyTorch is needed only to retrain the denoisers
python scripts/make_figures.py             # redraw all 15 figures of the paper from results/ (a few minutes)
python -m pytest -q tests                  # identity and sampler tests (~15 s)
bash scripts/build_paper.sh                # compile paper/main.pdf (needs pdflatex and bibtex)
```
Nothing is recomputed by these commands: they read the archived data in `results/`.

## Notebooks (executed, in `notebooks/`)

| Notebook | Content | Run time |
|---|---|---|
| `01_theory_checks` | bridge coefficients and the identities κa = γ, κ²σ² = γ, γ' = κ², Riccati; converse of Proposition 1 (protocol from a schedule); the field process on the clock γ (Monte Carlo); constants 0.149, 1.044, 3.76 of the binary-channel prediction; the bound Δ ≤ λ_P | seconds |
| `02_one_instance_from_scratch` | the whole analysis of the paper for one N = 16 Bethe glass: exact measure, classes, predicted decision times, trajectories generated endpoint first, partial commitments p_ℓ and q_ℓ, commitments and events, information per level, comparison with the prediction | < 1 min |
| `03_reproduce_figures` | runs `scripts/make_figures.py` and shows all figures in paper order | ~3 min |

Rebuild and re-execute them with `python scripts/build_notebooks.py`.

## Layout

| Folder | Content |
|---|---|
| `paper/` | `main.tex`, `refs.bib`, `main.bbl`, `main.pdf`, `figures/` (PDF and PNG, named in paper order: `Fig01`–`Fig11`, `FigA1`–`FigA4`) |
| `code/` | `exacttree.py`: exact enumeration up to N = 24 (half-split representation, compiled field kernel), classes (Sec. 4), endpoint-first trajectories, exact posteriors, information per level, parent spectra. `pidlib.py`, `pidlib_lite.py`: harmonic-PID clocks, Monte Carlo and sequential Monte Carlo samplers, the positive control |
| `experiments/` | scripts that generate the data and scripts that draw the figures (see `docs/FIGURES.md`) |
| `results/` | archived data used by the paper (about 150 MB of JSON/NPZ) |
| `notebooks/` | three executed notebooks (above) |
| `scripts/` | `make_figures.py`, `build_notebooks.py`, `build_paper.sh`, `run_all_experiments.sh` |
| `tests/` | `pytest` tests of the identities and samplers; `check_clock_theory.py` |
| `docs/` | `FIGURES.md`: figure → script → data → data generator |

## Figures

| Paper figure | File | Figure script (`experiments/`) | Data (`results/`) |
|---|---|---|---|
| 1 classes, partial commitments and clocks | `Fig01_schematic` | `n2_schematic.py` | numbers of Fig. 2 |
| 2 one N = 24 instance in detail | `Fig02_anatomy` | `n1_figures.py anatomy` | `N1/*anatomy*` |
| 3 how a trajectory descends the tree | `Fig03_partial` | `n2_figures.py partial` | `N3`, `N2/paths_*` |
| 4 classes across temperature | `Fig04_temperature` | `n1_figures.py temp` | `N1S` |
| 5 across instances and sizes | `Fig05_sizes` | `n1_figures.py size` | `N1S`, `N1` |
| 6 the binary-channel prediction | `Fig06_law` | `n1_figures.py law` | `N1` |
| 7 sparse (Bethe) and dense (SK) glasses | `Fig07_families` | `n4_families.py` | `N1`, `N1S`, `N2`, `N4` |
| 8 protocol and discretization | `Fig08_protocol` | `n2_figures.py protocol` | `N2/disc_*` |
| 9 structural tests and controls, N = 32, 64 | `Fig09_controls` | `e5e_summarize.py` | `E5E`, `E5F`, `S8` |
| 10 larger systems, N = 32, 64 | `Fig10_larger` | `e5d_analyze.py` | `E5B`–`E5D` |
| 11 hidden units of learned denoisers | `Fig11_learned` | `e6_figures_b.py` | `E6/E6B`, `S7` |
| A1, A2 positive control with a known tree | `FigA1_E0_model`, `FigA2_E0` | `fig_E0_model.py`, `fig_E0.py` | `E0`, `S8` |
| A3 contrast identities as numerical tools | `FigA3_tools` | `e6_figures_a.py` | `E6/E6A` |
| A4 label-free spectral prediction | `FigA4_static_law` | `s7_figure.py` | `S7`, `E5F` |

## Recomputing the data

`scripts/run_all_experiments.sh` lists every data-generating step in execution order with approximate run times (two cores):

1. **Exact campaign, N ≤ 24** (Figs. 2–8, about seven hours): `n1_run_all.sh`, `n1_run_extension.sh`, `n2_run.sh`, `n2_paths.py`,
   `n2_discretization.py`, `n3_partial.py`, `n4_run_sk.sh`.
2. **Positive control** (Appendix B, minutes): `e0_caricature.py`, `s8_validation.py`.
3. **Exact N = 16 cohort for the label-free prediction and contrast identities** (minutes): `e5_exact_transients.py`, `s7_exact16_*.py`.
4. **Sampled posteriors at N = 32, 64** (Figs. 9, 10, hours): `e5b_*`, `e5c_*`, `e5d_analyze.py`, `e5e_*`, `s7_e9_larger.py`.
5. **Contrast identities and learned denoisers** (Fig. 11, Appendix C, hours; PyTorch): `e6_*`, `s7_e10_learned.py`.

All random seeds are fixed in the scripts. Instances are unselected consecutive disorder seeds for N ≤ 24; the three N = 32, 64
instances per size were selected for reproducible classes (see the paper).

## Requirements

Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`, `numba`, `networkx`, `scikit-learn`; `torch` only for training the denoisers;
`jupyter`, `nbformat`, `nbclient`, `pytest` for notebooks and tests. Tested with Python 3.13, numpy 2.5, scipy 1.18, numba 0.68,
matplotlib 3.11.

## Citation

See `CITATION.cff`. Until the arXiv identifier is available:
```
@misc{chertkov2026pidspinglass,
  author = {Chertkov, Michael},
  title  = {How Path Integral Diffusion Decides a Finite Spin Glass: Stochastic Localization, Level-Resolved Information
            and Spectral Decision Times},
  year   = {2026},
  note   = {Code: https://github.com/mchertkov/SpinGlassHierarchy}
}
```

## Use of language models

A large language model was used to proof-read the manuscript and to help design, implement and check the numerical experiments.
All results were reviewed by the author, who is solely responsible for the content, the claims and any errors.

## License

MIT (see `LICENSE`).
