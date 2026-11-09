"""Figure: the solvable positive-control model E0 (mixture of 36 product states on a known tree, N=500).
Uses the same generator and seed as experiments/e0_caricature.py.  Writes results/E0/figures/E0_model.{png,pdf}."""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code"))
import pidlib as P

S = P.ultrametric_caricature(N=500); m, W, lab = S.m, S.W, S.labels; N = S.N
order = np.lexsort((lab[:, 3], lab[:, 2], lab[:, 1], lab[:, 0])); m, W, lab = m[order], W[order], lab[order]
Q = m @ m.T / N
fig = plt.figure(figsize=(15.5, 4.8)); gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.15, 1.05, 0.28], wspace=0.32)

# (a) generative tree
a = fig.add_subplot(gs[0, 0]); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
a.text(0.0, 0.985, "latent state $\\alpha$ = leaf;  $y\\mid\\alpha\\sim\\prod_i(1+m_{\\alpha i}y_i)/2$", ha="left", va="top", fontsize=8.2)
levels = [(0.76, "sign: $m_\\alpha$ or $-m_\\alpha$ (mirror)", 2), (0.56, "level 1: 2 independent random patterns", 2),
          (0.36, "level 2: 3 children, flip 15% of signs", 3), (0.16, "level 3: 3 children, flip 5% of signs", 3)]
xs = {0: [0.45]}; ys = {0: 0.90}
a.scatter([0.45], [0.90], s=30, color="k", zorder=3)
for depth, (y, txt, b) in enumerate(levels, start=1):
    prev = xs[depth - 1]; width = 0.86 / (len(prev) * b); cur = []
    for k, x0 in enumerate(prev):
        for c in range(b):
            x = 0.02 + width * (k * b + c + 0.5); cur.append(x)
            a.plot([x0, x], [ys[depth - 1], y], color="0.55", lw=0.5 if depth >= 3 else 0.9)
    xs[depth] = cur; ys[depth] = y
    a.scatter(cur, [y] * len(cur), s=8 if depth >= 3 else 20, color=["tab:blue", "tab:green", "tab:orange", "tab:red"][depth - 1], zorder=3)
    a.text(1.02, y + 0.035, txt, ha="right", fontsize=7.2)
a.text(0.45, 0.02, "$2\\times2\\times3\\times3=36$ states, magnetisation $|m_{\\alpha i}|=0.9$, weights: Dirichlet(1) at each branching", ha="center", fontsize=7.2)
a.set_title("(a) how the 36 states are generated", fontsize=9.5)

# (b) patterns
b = fig.add_subplot(gs[0, 1]); b.imshow(np.sign(m[:, :90]), cmap="coolwarm", aspect="auto", interpolation="nearest")
b.set_xlabel("spin $i$ (first 90 of 500)"); b.set_ylabel("state $\\alpha$ (ordered by tree)"); b.set_title("(b) sign patterns $\\mathrm{sign}(m_{\\alpha i})$", fontsize=9.5)

# (c) overlap matrix and (d) weights
c = fig.add_subplot(gs[0, 2]); im = c.imshow(Q, cmap="RdBu_r", vmin=-0.81, vmax=0.81)
c.set_xlabel("state $\\beta$"); c.set_ylabel("state $\\alpha$"); c.set_title("(c) state overlaps $q_{\\alpha\\beta}=m_\\alpha\\!\\cdot m_\\beta/N$", fontsize=9.5)
cb = plt.colorbar(im, ax=c, fraction=0.046); cb.ax.tick_params(labelsize=7)
d = fig.add_subplot(gs[0, 3], sharey=c); d.barh(np.arange(len(W)), W, color=np.where(lab[:, 0] == 0, "tab:blue", "tab:red"))
d.set_xlabel("weight $W_\\alpha$"); d.tick_params(labelleft=False); d.set_title("(d)", fontsize=9.5)
out = ROOT / "results/E0/figures"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "E0_model.png", dpi=160, bbox_inches="tight"); fig.savefig(out / "E0_model.pdf", bbox_inches="tight")
vals = sorted(set(np.round(Q[np.triu_indices(len(Q), 1)], 2)))
print("distinct overlaps (rounded):", [v for v in vals if abs(v) > 0.2][:8], "...", "weights min/max", W.min().round(4), W.max().round(3))
