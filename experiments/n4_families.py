"""Campaign N4 (2026-11-01): sparse (Bethe) versus dense (SK) glasses, compared at the same distance below the glass transition.
(a) prevalence of collective glassy levels, (b) split variance of the first glassy split, (c) commitments and distinct events per trajectory,
(d) fraction of trajectories whose sector and first glassy commitments are at least a factor two apart, (e) measured / predicted decision time of
the glassy levels, (f) the same for the first glassy level against the alignment Delta / lambda_P of the split with the parent's leading mode.
usage: python experiments/n4_families.py   -> paper/figures/Fig07_families.pdf, results/N4/families_summary.json
"""
from pathlib import Path
import sys, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'experiments'))
from n2_figures import recount
R1, RS, R2 = ROOT / 'results' / 'N1', ROOT / 'results' / 'N1S', ROOT / 'results' / 'N2'
OUT = ROOT / 'results' / 'N4'; OUT.mkdir(parents=True, exist_ok=True); FIG = ROOT / 'paper' / 'figures'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
BC = dict(bethe=1.34, sk=1.0)
BGRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
STY = {('bethe', 16): ('o', '#1f4e79'), ('bethe', 20): ('s', '#2e75b6'), ('sk', 16): ('^', '#c55a11'), ('sk', 20): ('v', '#e69a5c')}
NAME = dict(bethe='Bethe', sk='SK')


def jl(p): return json.load(open(p))


def prevalence(fam, N):
    xb, f1 = [], []
    for b in BGRID:
        fs = [f for f in glob.glob(str(RS / f'{fam}_N{N}_b{b}_s*.json'))]
        if len(fs) < 10: continue
        xb.append(b / BC[fam]); f1.append(np.mean([jl(f)['n_levels_glassy'] >= 1 for f in fs]))
    return xb, f1


def strength(fam, N):
    xb, d1 = [], []
    for b in BGRID:
        fs = glob.glob(str(RS / f'{fam}_N{N}_b{b}_s*.json'))
        if len(fs) < 10: continue
        k = [[c['kappa'] for c in jl(f)['candidates'] if c['level'] == 1] for f in fs]
        xb.append(b / BC[fam]); d1.append(np.median([x[0] / N for x in k if x]))
    return xb, d1


def alignment():
    """first glassy level: split variance over parent's leading eigenvalue, against measured / predicted decision time"""
    out = {}
    for fam in ('bethe', 'sk'):
        X, Y = [], []
        for f in glob.glob(str(R1 / f'{fam}_N*_b*_s*.json')):
            if 'anatomy' in f: continue
            o = jl(f)
            if o['n_levels_glassy'] < 1: continue
            acc = [c for c in o['candidates'] if c.get('accepted') and c['level'] == 1]
            ch = o['channels'][1] if len(o['channels']) > 1 else []
            hg, pc = o['half_gamma'], o['pred_cond']
            if acc and ch and len(hg) > 1 and hg[1] and pc[1]:
                X.append(acc[0]['kappa'] / max(c['kappa'] for c in ch)); Y.append(hg[1] / pc[1])
        out[fam] = (np.array(X), np.array(Y))
    return out


def per_path(fam, N):
    xb, nc, ne, far = [], [], [], []
    for b in BGRID:
        fs = sorted(glob.glob(str(R2 / f'paths_{fam}_N{N}_b{b}_s*.json')))
        C, E, F = [], [], []
        for f in fs:
            o = jl(f)
            if o['n_levels_glassy'] < 1: continue
            z = np.load(f.replace('.json', '.npz')); c, e, rat = recount(z); w = np.array(o['w']); w = w / w.sum()
            C.append(w @ c); E.append(w @ e)
            r1 = [r for l, r in rat if l == 1]
            if r1: F.append(np.mean(np.array(r1) >= 2))
        if len(C) >= 3: xb.append(b / BC[fam]); nc.append(np.mean(C)); ne.append(np.mean(E)); far.append(np.mean(F))
    return xb, nc, ne, far


def law(fam, N):
    xb, med, lo, hi, n = [], [], [], [], []
    for b in BGRID:
        r = []
        for f in glob.glob(str(R1 / f'{fam}_N{N}_b{b}_s*.json')):
            if 'anatomy' in f: continue
            o = jl(f); hg, pc = o['half_gamma'], o['pred_cond']
            for l in range(1, min(len(hg), len(pc))):
                if hg[l] and pc[l]: r.append(hg[l] / pc[l])
        if len(r) >= 5: xb.append(b / BC[fam]); q = np.percentile(r, [25, 50, 75]); lo.append(q[0]); med.append(q[1]); hi.append(q[2]); n.append(len(r))
    return xb, med, lo, hi, n


def main():
    from scipy.stats import spearmanr
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.6)); axs = axs.ravel(); S = {}
    for (fam, N), (mk, col) in STY.items():
        lab = f'{NAME[fam]} $N={N}$'; d = {}
        x, f1 = prevalence(fam, N); axs[0].plot(x, f1, '-' + mk, color=col, ms=4, label=lab); d['prevalence'] = dict(x=x, f=f1)
        x, d1 = strength(fam, N); axs[1].plot(x, d1, '-' + mk, color=col, ms=4); d['delta1_over_N'] = dict(x=x, d=d1)
        x, nc, ne, far = per_path(fam, N)
        if x:
            axs[2].plot(x, ne, '-' + mk, color=col, ms=4); axs[2].plot(x, nc, ':' + mk, color=col, ms=3, mfc='none')
            axs[3].plot(x, far, '-' + mk, color=col, ms=4)
        d['per_path'] = dict(x=x, commitments=nc, events=ne, sector_glassy_far=far)
        x, med, lo, hi, n = law(fam, N)
        if x:
            xo = np.array(x) * (1.01 if fam == 'sk' else 0.99)
            axs[4].errorbar(xo, med, yerr=[np.array(med) - lo, np.array(hi) - med], fmt='-' + mk, color=col, ms=4, capsize=2, lw=1)
        d['law_glassy'] = dict(x=x, median=med, q25=lo, q75=hi, n=n); S[f'{fam}_N{N}'] = d
    A = alignment(); ax = axs[5]; bins = [0.5, 0.7, 0.8, 0.9, 1.0]
    for fam, mk, col in [('bethe', 'o', '#1f4e79'), ('sk', '^', '#c55a11')]:
        X, Y = A[fam]; ax.semilogy(X, Y, mk, ms=2.5, alpha=.35, color=col, label=f'{NAME[fam]} ({len(X)})')
    X = np.concatenate([A[f][0] for f in A]); Y = np.concatenate([A[f][1] for f in A]); bm, bx = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (X >= lo) & (X < hi + 1e-9)
        if m.sum() >= 5: bx.append(np.median(X[m])); bm.append(np.median(Y[m]))
    ax.plot(bx, bm, 'k-s', ms=4, label='median (both)'); rho = spearmanr(X, Y)[0]
    S['alignment'] = dict(n=int(len(X)), spearman=float(rho), bin_x=bx, bin_median=bm,
                          bins={f'{lo}-{hi}': dict(n=int(((X >= lo) & (X < hi + 1e-9)).sum()), median=float(np.median(Y[(X >= lo) & (X < hi + 1e-9)])) if ((X >= lo) & (X < hi + 1e-9)).sum() else None) for lo, hi in zip(bins[:-1], bins[1:])})
    ax.set_xlabel('alignment $\\Delta/\\lambda_P$ of the first glassy split'); ax.set_title(f'(f) the prediction needs a split along the leading\nmode (Spearman {rho:.2f})')
    ax.legend(frameon=False, fontsize=7)
    axs[0].set_title('(a) instances with collective glassy levels'); axs[0].set_ylim(-0.02, 1.02)
    axs[1].set_title('(b) split variance of the best first\nglassy split, $\\Delta_1/N$ (median)')
    axs[2].set_title('(c) per trajectory: distinct events (solid)\nand commitments (dotted)'); axs[2].set_ylim(0.9, 3.2)
    axs[3].set_title('(d) trajectories with sector and first glassy\ncommitments at least a factor 2 apart'); axs[3].set_ylim(0, 1)
    axs[4].set_title('(e) glassy levels: measured / predicted\ndecision time (median, quartiles)')
    for a in (axs[4], axs[5]):
        a.set_yscale('log'); a.axhline(1, color='0.5', lw=.7); a.axhspan(1 / 1.3, 1.3, color='0.92', zorder=0)
        a.set_yticks([0.5, 0.7, 1, 1.5, 2, 3]); a.set_yticklabels(['0.5', '0.7', '1', '1.5', '2', '3']); a.minorticks_off()
    axs[4].set_ylim(0.6, 2.2); axs[5].set_ylim(0.4, 3.2)
    for a in axs[:5]: a.set_xlabel('$\\beta_J/\\beta_c$'); a.axvline(1, color='0.6', lw=.7, ls='--')
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'Fig07_families.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Fig07_families.png', dpi=150, bbox_inches='tight')
    json.dump(S, open(OUT / 'families_summary.json', 'w'), indent=1, default=float); print(json.dumps(S, default=float)[:3000])


if __name__ == '__main__':
    main()
