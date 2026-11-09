"""Schematic Fig. 1: classes (target) vs decisions (trajectory) vs clocks (protocol).
Numbers in (b, c) are those of the N=24 Bethe instance of the anatomy figure (sector lambda_1 = 13.5; glassy kappa_P = 4.9, 4.0, 3.4).
usage: python experiments/n2_schematic.py  -> paper/figures/Fig01_schematic.pdf
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / 'paper' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.spines.top': False, 'axes.spines.right': False})
LC = ['#1f4e79', '#c55a11', '#2e8b57', '#7f3f98']
xs = np.linspace(-8, 8, 2001); wx = np.exp(-xs ** 2 / 2); wx /= wx.sum()
def Ibin(s):
    s = np.atleast_1d(s)[:, None]; u = s + np.sqrt(s) * xs[None, :]
    return (np.log(2) - (np.logaddexp(0, -2 * u) * wx).sum(1)) / np.log(2)
def gam(S, t):
    return t / (1 - t) if S == 0 else S * np.sinh(S * t) / (np.sinh(S) * np.sinh(S * (1 - t)))

fig = plt.figure(figsize=(13, 4.3)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 1.0], wspace=.28)

# (a) landscape with mirror halves and nested valleys
ax = fig.add_subplot(gs[0]); ax.set_title('(a) classes: a property of the target $\\mu$')
cent = np.array([-7, -5.6, -3.6, -2.2, 2.2, 3.6, 5.6, 7]); depth = np.array([1.0, .8, .9, .7, .7, .9, .8, 1.0])
x = np.linspace(-8.5, 8.5, 1200); E = 1.6 - sum(d * np.exp(-(x - c) ** 2 / .35) for c, d in zip(cent, depth)) + 0.05 * x ** 2 / 8
ax.plot(x, E, 'k-', lw=1.2); ax.set_xlim(-11.5, 9); ax.set_ylim(-0.4, 4.6); ax.axis('off')
ax.axvline(0, color='0.6', ls=':', lw=.8); ax.text(0, -0.35, 'mirror: $y\\to-y$', ha='center', fontsize=7.5, color='0.4')
ax.text(-4.6, 1.85, 'sector $+$', ha='center', fontsize=8, color=LC[0]); ax.text(4.6, 1.85, 'sector $-$', ha='center', fontsize=8, color=LC[0])
# tree above the landscape (one sector drawn fully, mirror greyed)
def node(xx, yy, col, s=60): ax.scatter(xx, yy, s=s, color=col, zorder=3, edgecolor='white', lw=.6)
def edge(x1, y1, x2, y2, col='0.55'): ax.plot([x1, x2], [y1, y2], color=col, lw=.9, zorder=1)
yr, y0, y1, y2 = 4.3, 3.55, 2.9, 2.35
node(0, yr, '0.3', 90); ax.text(0.35, yr + 0.05, 'all configurations', fontsize=7)
for sgn, alpha in [(-1, 1), (1, .35)]:
    xs0 = sgn * 4.6; edge(0, yr, xs0, y0); node(xs0, y0, LC[0])
    g1 = [sgn * 6.3, sgn * 2.9]
    for xg in g1: edge(xs0, y0, xg, y1, LC[1] if alpha == 1 else '0.8'); node(xg, y1, LC[1] if alpha == 1 else '#e6b8a2')
    for xg, pair in zip(g1, [(sgn * 7, sgn * 5.6), (sgn * 3.6, sgn * 2.2)]):
        for xv in pair:
            edge(xg, y1, xv, y2, LC[2] if alpha == 1 else '0.85'); node(xv, y2, LC[2] if alpha == 1 else '#b4d9c3', 40)
ax.text(-9.0, y0, 'sector', fontsize=7, color=LC[0], va='center', ha='right'); ax.text(-9.0, y1, 'glassy 1', fontsize=7, color=LC[1], va='center', ha='right')
ax.text(-9.0, y2, 'glassy 2', fontsize=7, color=LC[2], va='center', ha='right')
ax.text(-4.6, -0.05, 'valleys = groups of\nlow-energy configurations', ha='center', fontsize=7, color='0.35')

# (b) one trajectory on the internal clock
ax = fig.add_subplot(gs[1]); ax.set_title('(b) when each split is resolved, on the clock $\\gamma$')
g = np.geomspace(1e-3, 30, 400)
kap = [(13.5, 'sector (mirror half)'), (4.9, 'glassy level 1'), (4.0, 'glassy level 2')]
for l, (k, lab) in enumerate(kap):
    ax.plot(g, Ibin(g * k), color=LC[l], lw=2, label=lab)
    ax.plot([0.149 / k, 3.76 / k], [-0.0 + 0.035 * (l + 1)] * 2, color=LC[l], lw=2.5, solid_capstyle='butt')
ax.plot(g, Ibin(g * 0.8), color='0.5', lw=1.2, ls='--', label='individual spins ($\\lambda_P\\lesssim1$)')
ax.set_xscale('log'); ax.set_xlim(1e-3, 30); ax.set_ylim(-0.02, 1.38); ax.set_xlabel('internal clock $\\gamma$ (signal-to-noise ratio)')
ax.set_ylabel('fraction of the level decided')
ax.annotate('', xy=(0.75, 1.07), xytext=(0.03, 1.07), arrowprops=dict(arrowstyle='<->', color=LC[1], lw=.9))
ax.text(0.15, 1.09, 'compound glassy decision', ha='center', fontsize=7, color=LC[1], va='bottom')
for xx, txt in [(1.5e-3, 'all'), (0.05, 'one half'), (0.9, 'one valley'), (12, 'one\nconfig.')]:
    ax.text(xx, 1.17, txt, fontsize=6.5, color='0.3', va='bottom', ha='left' if xx < 1e-2 else 'center')
ax.text(1.5e-3, 1.27, 'reachable endpoints of the trajectory:', fontsize=6.6, color='0.3', va='bottom')
ax.legend(frameon=False, fontsize=6.8, loc='center left', bbox_to_anchor=(0.0, 0.52))
ax.text(1.1, 0.16, 'bars: decision windows\n(10 to 90% decided);\neach spans a factor 25', fontsize=6.4, color='0.35')

# (c) protocol: same gamma windows, different physical times
ax = fig.add_subplot(gs[2]); ax.set_title('(c) the protocol $\\nu(t)$ changes when, not what')
t = np.linspace(1e-4, 0.999, 600)
for S, ls, lab in [(0, '-', '$\\nu=0$'), (4.9, '--', '$\\nu=24$ (tuned)'), (12, ':', '$\\nu=144$')]:
    ax.semilogy(t, gam(S, t), 'k', ls=ls, lw=1.3, label=lab)
for l, (k, _) in enumerate(kap[:2]):
    ax.axhspan(0.149 / k, 3.76 / k, color=LC[l], alpha=.12)
for row, (S, ls) in enumerate([(0, '-'), (4.9, '--'), (12, ':')]):
    for l, (k, _) in enumerate(kap[:2]):
        lo, hi = 0.149 / k, 3.76 / k
        ta = t[np.argmax(gam(S, t) >= lo)]; tb = t[np.argmax(gam(S, t) >= hi)]
        ax.plot([ta, tb], [1.6e-3 * 2.6 ** row * (1.3 if l else 1)] * 2, color=LC[l], lw=2.5, solid_capstyle='butt')
    ax.text(0.995, 1.75e-3 * 2.6 ** row, ['$\\nu=0$', 'tuned', 'stiff'][row], fontsize=6.5, ha='right', va='center', color='0.3')
ax.set_ylim(1e-3, 1e2); ax.set_xlim(0, 1); ax.set_xlabel('physical time $t$'); ax.set_ylabel('$\\gamma(t)$')
ax.legend(frameon=False, fontsize=7, loc='upper left'); ax.text(0.5, 3.5e-2, 'shaded: sector and glassy-1 windows in $\\gamma$\nbars below: the same windows in $t$', ha='center', fontsize=6.6, color='0.35')
fig.savefig(FIG / 'Fig01_schematic.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Fig01_schematic.png', dpi=150, bbox_inches='tight')
