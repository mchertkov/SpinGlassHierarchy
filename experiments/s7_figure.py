"""Stage 7 figure: static spectral law (binary-channel collapse) on the frozen exact cohort, larger-N check, learned denoisers."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; S7 = ROOT / "results/S7"; S_STAR = 1.042; (S7 / "figures").mkdir(parents=True, exist_ok=True)
z = np.random.default_rng(0).standard_normal(400000); sg = np.geomspace(1e-3, 60, 120)
Ib = np.array([1 - np.mean(np.logaddexp(0, -2 * (s + np.sqrt(s) * z))) / np.log(2) for s in sg])     # I(s)/ln 2
law = json.load(open(S7 / "exact16_binary_channel_law.json"))
fig, ax = plt.subplots(2, 3, figsize=(16, 9.6))
style = {("bethe", True): dict(c="tab:blue", mfc="tab:blue"), ("bethe", False): dict(c="tab:blue", mfc="none"),
         ("two_state", False): dict(c="tab:orange", mfc="tab:orange"), ("product", False): dict(c="0.5", mfc="0.5")}
a = ax[0, 0]; gg = np.geomspace(0.02, 30, 10)
for o in law:
    st = style[(o["kind"], o["acc"])]
    a.loglog(o["pred_c"], o["half_c"], "o", color=st["c"], mfc=st["mfc"], ms=7)
    if o["kind"] == "bethe": a.loglog(o["pred_f"], o["half_f"], "s", color=st["c"], mfc=st["mfc"], ms=7)
a.loglog(gg, gg, "k:", lw=0.8); a.fill_between(gg, gg / 1.3, gg * 1.3, color="0.85", zorder=0)
for lab, st, mk in [("Bethe, accepted tree", style[("bethe", True)], "o"), ("Bethe, not accepted", style[("bethe", False)], "o"),
                    ("two-state null", style[("two_state", False)], "o"), ("product null", style[("product", False)], "o")]:
    a.plot([], [], mk, color=st["c"], mfc=st["mfc"], label=lab)
a.plot([], [], "ks", mfc="none", label="fine level (squares), vs $s^*/\\lambda_2$")
a.set_xlabel(r"static prediction $s^*/\lambda_k$  ($s^*=1.04$)"); a.set_ylabel(r"measured half-information $\gamma$"); a.legend(fontsize=6.5)
a.set_title("(a) exact $N=16$ cohort: binary-channel prediction", fontsize=9)
for col, (lvl, ttl) in enumerate([(0, "(b) coarse level: collapse on $s=\\gamma\\lambda_1$"), (1, "(c) fine level: collapse on $s=\\gamma\\lambda_2$")]):
    a = ax[0, 1 + col]
    for f in sorted(glob.glob(str(ROOT / "results/E5F/exact16/*.json"))):
        d = json.load(open(f)); kind = d["kind"]
        if kind == "product" or (lvl == 1 and kind != "bethe"): continue
        x = np.array(d["information"]); g = np.array(d["gamma"]); lam = json.load(open(S7 / ("exact16_" + Path(f).name)))["lam"]
        st = style[(kind, bool(d["accepted_candidate_tree"]))]; ls = "-" if st["mfc"] != "none" else "--"
        a.semilogx(g[1:] * lam[lvl], x[1:, lvl] / x[-1, lvl], ls, color=st["c"], lw=1, alpha=0.8)
    a.semilogx(sg, Ib, "k-", lw=2.5, label="binary channel $I(s)/\\log 2$"); a.axvline(S_STAR, color="k", ls=":")
    a.set_xlim(5e-3, 50); a.set_ylim(-0.05, 1.1); a.set_xlabel(r"$s=\gamma\lambda_%d$" % (lvl + 1)); a.set_ylabel("information fraction"); a.legend(fontsize=7); a.set_title(ttl, fontsize=9)
a = ax[1, 0]
for f in sorted(glob.glob(str(S7 / "E9_*.json"))):
    r = json.load(open(f)); c = "tab:blue" if "bethe" in f else "tab:green"; ms = 6 if r["N"] < 50 else 9; lam = r["lam"]; w = r["windows"]
    a.loglog(S_STAR / lam[0], w["coarse"]["half"], "o", color=c, ms=ms); a.loglog(S_STAR / lam[1], w["fine"]["half"], "s", color=c, ms=ms, mfc="none")
a.loglog(gg, gg, "k:", lw=0.8); a.fill_between(gg, gg / 1.3, gg * 1.3, color="0.85", zorder=0)
a.plot([], [], "o", color="tab:blue", label="Bethe"); a.plot([], [], "o", color="tab:green", label="planar"); a.plot([], [], "ks", mfc="none", label="fine (open)")
a.set_xlabel(r"$s^*/\lambda_k$"); a.set_ylabel("measured half-information"); a.legend(fontsize=7); a.set_title("(d) $N=32$-$64$ (small/large markers), new instances", fontsize=9)
cols = {"inf": "k", "1024": "tab:orange", "64": "tab:red"}
for M, c in cols.items():
    r = json.load(open(S7 / f"E10_M{M}.json")); g = np.array(r["gamma"])
    ax[1, 1].loglog(g, np.array(r["chi_th"]) / np.array(r["chi_ex"]), "o-", ms=3, color=c, label=f"training set M={M}")
    ax[1, 2].semilogx(g, np.array(r["mse_th"]) - np.array(r["mse_ex"]), "o-", ms=3, color=c, label=f"M={M}")
ref = json.load(open(S7 / "E7_bethe_s1.json"))
for a in ax[1, 1:]:
    a.axvline(ref["windows"]["coarse(Z2)"]["gamma_half"], color="tab:blue", ls=":"); a.axvline(ref["windows"]["fine|coarse"]["gamma_half"], color="tab:orange", ls=":")
ax[1, 1].axhline(1, color="0.5", lw=0.6); ax[1, 1].set_xlabel(r"$\gamma$"); ax[1, 1].set_ylabel(r"$\chi_\theta/\chi_{\rm SG}$"); ax[1, 1].legend(fontsize=7)
ax[1, 1].set_title(r"(e) learned denoiser: Jacobian susceptibility $\|\partial\hat y_\theta/\partial h\|_F^2/N$", fontsize=8)
ax[1, 2].axhline(0, color="0.5", lw=0.6); ax[1, 2].set_xlabel(r"$\gamma$"); ax[1, 2].set_ylabel("excess denoising error"); ax[1, 2].legend(fontsize=7)
ax[1, 2].set_title("(f) memorisation appears after the fine window", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT / "results/S7/figures/Fig7_static.png", dpi=150); fig.savefig(ROOT / "results/S7/figures/Fig7_static.pdf"); print("ok")
