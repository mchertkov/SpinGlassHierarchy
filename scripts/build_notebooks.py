"""Write and execute the three notebooks in notebooks/.

usage:  python scripts/build_notebooks.py            # all three (about four minutes, mostly notebook 03)
        python scripts/build_notebooks.py 01 02      # selected notebooks
"""
from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks'
SETUP = """import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'experiments')]
np.set_printoptions(precision=4, suppress=True)"""


def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS = {}

# --------------------------------------------------------------------------------------------------------------------------
NOTEBOOKS['01_theory_checks'] = [
    md("""
# 01 - Numerical checks of the theory

Quick checks (seconds) of the formulas of Sections 2-3 of the paper:

1. the harmonic-PID bridge coefficients and the identities $\\kappa a=\\gamma$, $\\kappa^2\\sigma^2=\\gamma$ (Eq. `field-snr`), $\\gamma'=\\kappa^2$,
   $\\kappa'=c\\kappa$ and the Riccati equation $c'=c^2-\\nu$;
2. the converse of Proposition 1: from a signal-to-noise schedule $\\gamma(t)$ back to the protocol $\\nu(t)$;
3. the field process at fixed endpoint is a Brownian motion with drift $y$ on the clock $\\gamma$ (Monte Carlo);
4. the constants of the binary-channel prediction, $s=0.149,\\ 1.044,\\ 3.76$ at $10\\%,50\\%,90\\%$ of $\\log 2$;
5. the split-variance bound $\\Delta\\le\\lambda_P$ (law of total variance) on random measures.
"""),
    code(SETUP),
    md("## 1. Harmonic PID coefficients (Eq. `hpid`)"),
    code("""
def hpid(t, nu):
    S = np.sqrt(nu)
    if S < 1e-9:
        a, s2, kap, c = t, t * (1 - t), 1 / (1 - t), 1 / (1 - t)
    else:
        a = np.sinh(S * t) / np.sinh(S); s2 = np.sinh(S * t) * np.sinh(S * (1 - t)) / (S * np.sinh(S))
        kap = S / np.sinh(S * (1 - t)); c = S / np.tanh(S * (1 - t))
    return a, s2, kap, c, a * a / s2

t = np.linspace(0.02, 0.98, 97); eps = 1e-6
deriv = lambda f: (f(t + eps) - f(t - eps)) / (2 * eps)          # symmetric difference with a tiny step
for nu in (0.0, 4.0, 13.0, 144.0):
    a, s2, kap, c, g = hpid(t, nu)
    G = lambda x: hpid(x, nu)[4]; Kp = lambda x: hpid(x, nu)[2]; Cc = lambda x: hpid(x, nu)[3]
    rel = lambda x, y: np.max(np.abs(x - y) / np.abs(y))
    print(f'nu={nu:6.1f}  kappa*a=gamma: {np.max(np.abs(kap*a-g)):.0e}   kappa^2 sigma^2=gamma: {np.max(np.abs(kap**2*s2-g)):.0e}'
          f"   gamma'=kappa^2: {rel(deriv(G), kap**2):.0e}   kappa'=c kappa: {rel(deriv(Kp), c*kap):.0e}   c'=c^2-nu: {np.max(np.abs(deriv(Cc) - (c**2 - nu)) / c**2):.0e}")
"""),
    md("""All five identities hold to the accuracy of the numerical derivative (relative errors of order $10^{-8}$ or smaller)."""),
    md("## 2. Converse of Proposition 1: protocol from a schedule"),
    code("""
# take the harmonic schedule for nu = 13 and recover nu(t) = c^2 - c', c = kappa'/kappa, kappa = sqrt(gamma')
nu_true = 13.0
tt = np.linspace(0.02, 0.98, 2001); h = tt[1] - tt[0]
g = hpid(tt, nu_true)[4]
kap = np.sqrt(np.gradient(g, h)); c = np.gradient(kap, h) / kap; nu_rec = c ** 2 - np.gradient(c, h)
inner = slice(50, -50)
print('recovered nu(t): median %.3f, range [%.3f, %.3f]  (true %.1f)' % (np.median(nu_rec[inner]), nu_rec[inner].min(), nu_rec[inner].max(), nu_true))
print('terminal condition (1-t)^2 gamma\\'(t) at t=0.98: %.3f (-> 1)' % ((1 - tt[-1]) ** 2 * np.gradient(g, h)[-1]))
"""),
    md("## 3. The field $h_t=\\kappa X_t$ given the endpoint is $\\gamma y$ plus noise of variance $\\gamma$"),
    code("""
rng = np.random.default_rng(1)
N, nu = 8, 4.0
y = rng.choice([-1.0, 1.0], N)
K = 400; tg = np.linspace(0, 1, K + 1)[:-1]; dtt = 1.0 / K; M = 20000
X = np.zeros((M, N)); checks = []
for k, tk in enumerate(tg):
    if k in (K // 4, K // 2, 3 * K // 4):
        a, s2, kap, c, g = hpid(np.array(tk), nu)
        hfield = kap * X
        checks.append((tk, float(g), float(np.mean(hfield @ y) / N), float(np.var(hfield - g * y))))
    a, s2, kap, c, g = hpid(np.array(tk), nu) if tk > 0 else (0, 0, np.sqrt(nu) / np.sinh(np.sqrt(nu)), np.sqrt(nu) / np.tanh(np.sqrt(nu)), 0)
    X = X + (kap * y - c * X) * dtt + np.sqrt(dtt) * rng.standard_normal((M, N))     # bridge drift kappa y - c x
for tk, g, mean_proj, var in checks:
    print(f't={tk:.2f}: gamma={g:.3f}   mean of h.y/N = {mean_proj:.3f}   variance of h - gamma y = {var:.3f}')
"""),
    md("Both the mean projection and the variance equal $\\gamma(t)$, up to Euler and sampling error."),
    md("## 4. Binary Gaussian channel and the constants of the prediction (Eq. `law`)"),
    code("""
xs, ws = np.polynomial.hermite_e.hermegauss(120); ws = ws / ws.sum()
def Ibin(s):
    s = np.atleast_1d(s)[:, None]
    return np.log(2) - (np.logaddexp(0, -2 * (s + np.sqrt(s) * xs[None, :])) * ws).sum(1)
from scipy.optimize import brentq
for frac in (0.1, 0.5, 0.9):
    print(f'I_bin(s) = {frac:.0%} of log 2 at s = {brentq(lambda s: Ibin(s)[0] / np.log(2) - frac, 1e-4, 50):.3f}')
ss = np.geomspace(1e-2, 1e2, 300)
plt.semilogx(ss, Ibin(ss) / np.log(2)); plt.xlabel('$s=\\\\gamma\\\\lambda_P$'); plt.ylabel('$I_{bin}(s)/\\\\log 2$'); plt.grid(alpha=.3); plt.show()
"""),
    md("## 5. Split variance is bounded by the parent's leading eigenvalue (Eq. `splitvar`)"),
    code("""
rng = np.random.default_rng(2); worst = 0
for trial in range(2000):
    n, N = 40, 6
    Ycfg = rng.choice([-1.0, 1.0], (n, N)); p = rng.dirichlet(np.ones(n)); side = rng.random(n) < 0.5
    if side.all() or (~side).all(): continue
    w1, w2 = p[side].sum(), p[~side].sum()
    m1, m2 = p[side] @ Ycfg[side] / w1, p[~side] @ Ycfg[~side] / w2
    Delta = w1 * w2 * np.sum((m1 - m2) ** 2)
    m = p @ Ycfg; C = (Ycfg * p[:, None]).T @ Ycfg - np.outer(m, m)
    worst = max(worst, Delta / np.linalg.eigvalsh(C)[-1])
print('max Delta / lambda_P over 2000 random measures and splits: %.4f (<= 1)' % worst)
"""),
]

# --------------------------------------------------------------------------------------------------------------------------
NOTEBOOKS['02_one_instance_from_scratch'] = [
    md("""
# 02 - One instance from scratch

The complete analysis of the paper (Table 1) for one $N=16$ Bethe glass at $\\beta_J=2.5$, computed from scratch in well under a minute:

1. the exact Gibbs measure (all $2^{16}$ configurations);
2. the classes (Sec. 4): sectors, glassy splits, their masses, overlaps and split variances $\\Delta$;
3. the predicted decision times from the parents' leading eigenvalues $\\lambda_P$ (binary-channel prediction);
4. trajectories generated endpoint first, $H_\\gamma=\\gamma Y+B_\\gamma$ (Proposition 1);
5. partial commitments $p_\\ell=p_{c_\\ell(Y)}$ and $q_\\ell=p_\\ell/p_{\\ell-1}$ along each trajectory, commitments and distinct events;
6. the information about each level, the measured decision times, and their comparison with the prediction.
"""),
    code(SETUP + "\nimport exacttree as E"),
    md("## 1. The target"),
    code("""
N, beta_J, seed = 16, 2.5, 9
J = E.bethe_J(N, beta_J, seed)                      # random three-regular graph, Gaussian couplings, J = beta_J J0
M = E.ExactModel(J, 'bethe_N16')
print('number of edges:', int((np.abs(J) > 0).sum() // 2), '  leading eigenvalue of C0: lambda_1 = %.2f' % M.lam[0])
"""),
    md("## 2. Classes (Sec. 4)"),
    code("""
T = E.build_tree(M)
print('classes per level (both sectors):', T['n_classes'])
for l, splits in enumerate(T['level_info'], start=1):
    for s in splits:
        print(f'level {l}: parent mass {s["mass"]:.3f} -> children {s["masses"][0]/s["mass"]:.2f} / {s["masses"][1]/s["mass"]:.2f} of the parent;'
              f'  overlap within {s["qin"]:.2f}, between {s["qbt"]:.2f};  split variance Delta = {s["kappa"]:.2f}')
"""),
    md("""Each split is a cut by a hyperplane in the space of aligned configurations (Eq. `hyperplane`). The children of the first glassy split
differ mainly on a block of spins on which their magnetizations have opposite signs:"""),
    code("""
s = M.gauge(); lab1 = T['labels'][1]
W = M.p * (s > 0); W = W / W.sum()                                    # one sector, aligned configurations
kids = [c for c in np.unique(lab1) if (W * (lab1 == c)).sum() > 1e-12]
mus = [M.first_moment(W * (lab1 == c)) / (W * (lab1 == c)).sum() for c in kids]
print('magnetizations of the two glassy classes (one row each):'); print(np.round(np.array(mus), 2))
print('spins with opposite signs:', np.flatnonzero(np.sign(mus[0]) != np.sign(mus[1])))
"""),
    md("## 3. Predicted decision times (binary-channel prediction)"),
    code("""
xs, ws = np.polynomial.hermite_e.hermegauss(120); ws = ws / ws.sum()
def Ibin_frac(s):
    s = np.atleast_1d(s)[:, None]
    return (np.log(2) - (np.logaddexp(0, -2 * (s + np.sqrt(s) * xs[None, :])) * ws).sum(1)) / np.log(2)
chans = E.parent_channels(M, T)             # per level: masses, entropies and leading eigenvalues lambda_P of the parents that split
gg = np.geomspace(1e-3, 20, 2000)
def predicted_curve(ch):
    w = np.array([c['pi'] * c['H'] for c in ch]); w = w / w.sum()
    return sum(wi * Ibin_frac(gg * c['kappa']) for wi, c in zip(w, ch))
pred = [float(gg[np.argmax(predicted_curve(ch) >= 0.5)]) for ch in chans]
for l, ch in enumerate(chans):
    print(f'level {l}: parents lambda_P = {[round(c["kappa"], 2) for c in ch]}  ->  predicted gamma_1/2 = {pred[l]:.3f}')
"""),
    md("## 4. Trajectories, generated endpoint first"),
    code("""
rng = np.random.default_rng(1)
g = np.concatenate([[0.0], np.geomspace(0.02 / N, 5.0, 30)])
R = E.run_paths(M, T, g, 256, rng)          # endpoints stratified over the finest classes, H = gamma Y + B, exact class posteriors
print('trajectories:', len(R['Y']), '  grid points:', len(g))
"""),
    md("## 5. Partial commitments along the endpoint's branch"),
    code("""
L = len(T['labels']); n = len(R['Y']); tau = N * g
p = np.stack([R['post'][l][np.arange(n), :, R['plant'][l]] for l in range(L)], -1)      # p_l = p_{c_l(Y)}
q = p.copy(); q[:, :, 1:] = p[:, :, 1:] / np.maximum(p[:, :, :-1], 1e-12)                 # q_l = p_l / p_{l-1}
i0 = 0
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for l in range(L):
    ax[0].semilogx(tau[1:], p[i0, 1:, l], label=f'level {l}'); ax[1].semilogx(tau[1:], np.median(q[:, 1:, l], 0), label=f'level {l}')
ax[0].axhline(.9, ls=':', c='k'); ax[0].set_title('one trajectory: $p_\\\\ell$'); ax[1].set_title('median over trajectories: $q_\\\\ell$')
for a in ax: a.set_xlabel('$\\\\tau=N\\\\gamma$'); a.legend(fontsize=8)
plt.show()
# commitments (first time p_l >= 0.9) at levels where the branch splits, merged into distinct events within a factor 2
split = np.ones((n, L), bool)
for l in range(1, L): split[:, l] = np.abs(p[:, 0, l] - p[:, 0, l - 1]) > 1e-12
ncom, nev = [], []
for i in range(n):
    tc = sorted(g[np.argmax(p[i, :, l] >= .9)] for l in range(L) if split[i, l] and (p[i, :, l] >= .9).any())
    ncom.append(len(tc)); nev.append(1 + sum(b / a >= 2 for a, b in zip(tc[:-1], tc[1:])) if tc else 0)
print('mean commitments per trajectory %.2f, mean distinct events %.2f' % (np.average(ncom, weights=R['w']), np.average(nev, weights=R['w'])))
"""),
    md("## 6. Information per level and the prediction"),
    code("""
I, dI, Hl, dH = E.level_information(M, T, R)
frac = np.array([dI[l] / dH[l] for l in range(L)])
meas = [E.half_time(g, frac[l]) for l in range(L)]
for l in range(L):
    print(f'level {l}: measured gamma_1/2 = {meas[l]:.3f}   predicted = {pred[l]:.3f}   ratio = {meas[l] / pred[l]:.2f}')
for l in range(L):
    plt.semilogx(g[1:], frac[l, 1:], 'o', ms=3, label=f'level {l} measured'); plt.semilogx(gg, predicted_curve(chans[l]), '--', color=plt.gca().lines[-1].get_color())
plt.xlabel('$\\\\gamma$'); plt.ylabel('information / final value'); plt.legend(fontsize=7); plt.xlim(1e-3, 5); plt.show()
"""),
]

# --------------------------------------------------------------------------------------------------------------------------
FIGLIST = ['Fig01_schematic', 'Fig02_anatomy', 'Fig03_partial', 'Fig04_temperature', 'Fig05_sizes', 'Fig06_law', 'Fig07_families',
           'Fig08_protocol', 'Fig09_controls', 'Fig10_larger', 'Fig11_learned', 'FigA1_E0_model', 'FigA2_E0', 'FigA3_tools', 'FigA4_static_law']
NOTEBOOKS['03_reproduce_figures'] = [
    md("""
# 03 - Reproduce all figures from the archived data

Runs `scripts/make_figures.py`, which calls the figure scripts in `experiments/` on the archived data in `results/` (nothing is
recomputed; about three minutes, most of it for Fig. 6) and writes `paper/figures/`. The figures are then displayed in paper order.
The table in `docs/FIGURES.md` maps each figure to its script, its data, and the scripts that generated the data.
"""),
    code(SETUP + "\nimport subprocess\nfrom IPython.display import Image, display, Markdown"),
    code("print(subprocess.run([sys.executable, str(ROOT / 'scripts' / 'make_figures.py')], cwd=ROOT, capture_output=True, text=True).stdout)"),
    code("for f in " + repr(FIGLIST) + ":\n    display(Markdown(f'**{f}**')); display(Image(filename=str(ROOT / 'paper' / 'figures' / (f + '.png')), width=900))"),
]


def main():
    sel = sys.argv[1:]
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        if sel and not any(name.startswith(s) for s in sel): continue
        nb = nbf.v4.new_notebook(); nb.cells = cells
        nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
        NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(NB)}}).execute()
        nbf.write(nb, NB / f'{name}.ipynb'); print('executed', name, flush=True)


if __name__ == '__main__':
    main()
