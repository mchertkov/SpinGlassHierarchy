"""Paper figure: solvable hierarchical positive control E0 (exact mixture of product states, N=500, 36 states).
Reads results/E0/E0.json and results/E0/E0_paths_beta0.npz; writes results/E0/figures/E0*.{png,pdf}."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "results/E0/figures"; FIG.mkdir(parents=True, exist_ok=True)


def e0():
    r = json.loads((ROOT / "results/E0/E0.json").read_text()); z = np.load(ROOT / "results/E0/E0_paths_beta0.npz"); run = r["runs"]["0.0"]
    lev = r["levels_selfoverlap"]; fig, ax = plt.subplots(2, 3, figsize=(16, 9)); cm = plt.get_cmap("viridis")
    t = z["t"]
    for k in range(40): ax[0, 0].semilogx(t, z["q"][k], color="0.3", lw=0.6, alpha=0.6)
    for v in lev.values(): ax[0, 0].axhline(v, color="tab:red", ls=":", lw=0.8)
    ax[0, 0].set_xlabel("t"); ax[0, 0].set_ylabel(r"$q(t)=|\hat y|^2/N$"); ax[0, 0].set_title(r"(a) order-parameter paths, $\nu=0$ (dotted: cluster levels)", fontsize=9)
    H = np.array([np.histogram(z["q"][:, k], bins=40, range=(0, 1))[0] for k in range(len(t))])
    ax[0, 1].imshow(np.log1p(H.T), origin="lower", aspect="auto", extent=[np.log10(t[0]), np.log10(t[-1]), 0, 1], cmap="magma")
    ax[0, 1].set_xlabel(r"$\log_{10} t$"); ax[0, 1].set_ylabel("q"); ax[0, 1].set_title("(b) distribution of q(t) over trajectories", fontsize=9)
    for i, (k, v) in enumerate(run["branching"].items()):
        c = cm(i / 4); ax[0, 2].hist(v["Q"], bins=40, range=(-1, 1), histtype="step", color=c, density=True, label=rf"$t_s$={v['t_s']:.3g}")
        ax[0, 2].axvline(v["mean_q_ts"], color=c, ls=":")
    ax[0, 2].set_xlabel(r"overlap $Q_{ab}$ of branched endpoints"); ax[0, 2].legend(fontsize=7); ax[0, 2].set_title(r"(c) genealogies: dotted = $\mathbb{E}\,q(t_s)$", fontsize=9)
    for i, (b, rr) in enumerate(r["runs"].items()):
        ax[1, 0].semilogx(rr["t"], rr["q_mean"], color=cm(i / 3), label=rf"$\nu$={float(b):g}")
        ax[1, 1].loglog(r["gammas"], rr["q_mean"], color=cm(i / 3), lw=3 - i * 0.6, label=rf"$\nu$={float(b):g}")
    ax[1, 0].set_xlabel("t"); ax[1, 0].set_ylabel(r"$\mathbb{E}\,q(t)$"); ax[1, 0].legend(fontsize=7); ax[1, 0].set_title("(d) stiffness schedules the decisions", fontsize=9)
    ax[1, 1].set_xlabel(r"$\gamma(t)$"); ax[1, 1].set_title(r"(e) collapse on the clock $\gamma$", fontsize=9); ax[1, 1].legend(fontsize=7)
    en = run["energy"]; ax[1, 2].plot(en["t"], en["energy"], "o", ms=3, label=r"$\frac{1}{2}\int_0^t \mathbb{E}|u|^2ds$")
    ax[1, 2].plot(en["t"], en["prediction"], "-", label=r"$-\frac{N}{2}\log(1-t)-I(y;x_t)$"); ax[1, 2].set_xlabel("t"); ax[1, 2].legend(fontsize=7)
    ax[1, 2].set_title(r"(f) energy--information identity, $\nu=0$", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "E0.png", dpi=150); fig.savefig(FIG / "E0.pdf")
    fig, a = plt.subplots(figsize=(6, 3.6))
    for i, (b, rr) in enumerate(r["runs"].items()):
        med = [rr["commit"][str(l)]["median"] for l in range(4)]; lo = [rr["commit"][str(l)]["q25"] for l in range(4)]; hi = [rr["commit"][str(l)]["q75"] for l in range(4)]
        a.errorbar(np.arange(4) + 0.08 * i, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="o-", color=cm(i / 3), label=rf"$\nu$={float(b):g}")
    a.set_yscale("log"); a.set_xticks(range(4)); a.set_xticklabels(["sign", "level 1", "level 2", "level 3"]); a.set_ylabel("commitment time t (90%)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "E0_commit.png", dpi=150); fig.savefig(FIG / "E0_commit.pdf")
    tri = run["triangles"]; Qab, Qac, Qbc = map(np.array, (tri["Q_ab"], tri["Q_ac"], tri["Q_bc"]))
    s = np.sort(np.abs(np.stack([Qab, Qac, Qbc], 1)), 1); nd = s[:, 2] - s[:, 0] > 0.02; K = (s[nd, 1] - s[nd, 0]) / (s[nd, 2] - s[nd, 0])
    return dict(triangle_means=[float(Qab.mean()), float(Qac.mean()), float(Qbc.mean())], triangle_expected=tri["expected"],
                frac_isosceles_K_lt_0p2=float(np.mean(K < 0.2)), energy_last=[en["energy"][-1], en["prediction"][-1]],
                commit={b: {l: rr["commit"][l]["median"] for l in rr["commit"]} for b, rr in r["runs"].items()},
                branching={k: [v["mean_Q"], v["mean_q_ts"]] for k, v in run["branching"].items()}, levels=lev)




if __name__ == "__main__":
    print(json.dumps(e0(), indent=1, default=float)[:600])
