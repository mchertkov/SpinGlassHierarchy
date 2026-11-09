"""Redraw every figure of the paper from the archived data in results/ (nothing is recomputed) into paper/figures/.

usage:  python scripts/make_figures.py            # all figures (about a minute)
        python scripts/make_figures.py Fig02 FigA2 # only figures whose names start with these prefixes
"""
from pathlib import Path
import shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
EXP, DEST = ROOT / 'experiments', ROOT / 'paper' / 'figures'
# (paper file name, command, file produced by the command; None = the command writes directly into paper/figures)
FIGS = [
    ('Fig01_schematic',   ['n2_schematic.py'],                None),
    ('Fig02_anatomy',     ['n1_figures.py', 'anatomy'],       None),
    ('Fig03_partial',     ['n2_figures.py', 'partial'],       None),
    ('Fig04_temperature', ['n1_figures.py', 'temp'],          None),
    ('Fig05_sizes',       ['n1_figures.py', 'size'],          None),
    ('Fig06_law',         ['n1_figures.py', 'law'],           None),
    ('Fig07_families',    ['n4_families.py'],                 None),
    ('Fig08_protocol',    ['n2_figures.py', 'protocol'],      None),
    ('Fig09_controls',    ['e5e_summarize.py'],               'results/E5E/figures/E5E_joint_structural_dynamic'),
    ('Fig10_larger',      ['e5d_analyze.py'],                 'results/E5D/figures/E5D_information_sensitivity'),
    ('Fig11_learned',     ['e6_figures_b.py'],                'results/E6/figures/E6B_learned_units'),
    ('FigA1_E0_model',    ['fig_E0_model.py'],                'results/E0/figures/E0_model'),
    ('FigA2_E0',          ['fig_E0.py'],                      'results/E0/figures/E0'),
    ('FigA3_tools',       ['e6_figures_a.py'],                'results/E6/figures/E6A_contrast_algorithms'),
    ('FigA4_static_law',  ['s7_figure.py'],                   'results/S7/figures/Fig7_static'),
]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    sel = sys.argv[1:]
    for name, cmd, produced in FIGS:
        if sel and not any(name.startswith(p) for p in sel): continue
        tic = time.time()
        subprocess.run([sys.executable, str(EXP / cmd[0]), *cmd[1:]], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        if produced:
            for ext in ('.pdf', '.png'):
                src = ROOT / (produced + ext)
                if src.is_file(): shutil.copy2(src, DEST / (name + ext))
        if not (DEST / (name + '.pdf')).is_file(): raise RuntimeError(f'{name}: no output')
        print(f'{name:18s} <- experiments/{cmd[0]:20s} {time.time() - tic:5.1f}s', flush=True)


if __name__ == '__main__':
    main()
