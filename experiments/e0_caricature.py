"""E0 -- exactly solvable ultrametric caricature in H-PID time.
Outputs: results/E0.json, results/E0.npz.   Runtime ~3 min."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code"))
import pidlib as P

t0 = time.time(); rng = np.random.default_rng(2026)
S = P.ultrametric_caricature(N=500); N = S.N; lab = S.labels                     # labels: (mirror, l1, l2, l3)
Qss = S.m @ S.m.T / N; qEA = float(np.diag(Qss).mean())
BETAS = [0.0, 4.0, 16.0, 36.0]; GAM = np.geomspace(1e-4, 30, 44); n_traj = 300
a_idx = rng.choice(S.A, n_traj, p=S.W)
Y = np.where(rng.random((n_traj, N)) < (1 + S.m[a_idx]) / 2, 1.0, -1.0)

def cluster_masses(post, a_true, level):
    """posterior mass on the true cluster at a given tree level (0 = mirror family, 1..3 = tree levels)."""
    key = lab[:, : level + 1]; same = (key[None, :, :] == key[a_true][:, None, :]).all(-1)
    return (post * same).sum(1)

res = dict(N=N, qEA=qEA, levels_selfoverlap={}, betas=BETAS, gammas=GAM.tolist(), runs={})
# cluster-average self-overlaps (where q(t) paths should plateau)
for lev in range(4):
    vals = []
    for a in range(S.A):
        key = lab[:, : lev + 1]; same = (key == key[a]).all(1); w = S.W[same] / S.W[same].sum()
        mm = (w[:, None] * S.m[same]).sum(0); vals.append(mm @ mm / N)
    res["levels_selfoverlap"][str(lev)] = float(np.average(vals, weights=S.W))
for beta in BETAS:
    B = P.Bridge(beta); tg = B.t_of_gamma(GAM)
    r = P.order_parameter_paths(S, B, Y, tg, rng, n_sub=20)
    commit = {}
    for lev in range(4):
        tau = np.full(n_traj, np.nan)
        for k, t in enumerate(tg):
            mass = cluster_masses(S.post_states(B.kappa(t) * r["X"][:, k]), a_idx, lev)
            new = np.isnan(tau) & (mass >= 0.9); tau[new] = t
        commit[str(lev)] = dict(median=float(np.nanmedian(tau)), q25=float(np.nanquantile(tau, .25)), q75=float(np.nanquantile(tau, .75)),
                                false_rate=None)
    run = dict(t=tg.tolist(), q_mean=r["q"].mean(0).tolist(), q_truth_mean=r["q_truth"].mean(0).tolist(), commit=commit)
    if beta == 0.0:
        # branching genealogies at several split times
        br = {}
        for g in (0.003, 0.01, 0.03, 0.1, 1.0):
            k = int(np.argmin(abs(GAM - g))); Qo, Qbb, _ = P.branching_overlaps(S, B, r["X"][:, k], tg[k], Y, 2, rng)
            br[f"{GAM[k]:.4g}"] = dict(t_s=float(tg[k]), mean_Q=float(Qo.mean()), mean_q_ts=float(r["q"][:, k].mean()), Q=Qo[:, 0].tolist())
        run["branching"] = br
        # three-way genealogy: c splits at t1, b splits at t2 > t1 from trajectory a
        k1, k2 = int(np.argmin(abs(GAM - 0.01))), int(np.argmin(abs(GAM - 0.1)))
        yc = S.sample(B.kappa(tg[k1]) * r["X"][:, k1], 1, rng)[:, 0]; yb = S.sample(B.kappa(tg[k2]) * r["X"][:, k2], 1, rng)[:, 0]
        tri = np.stack([(Y * yb).sum(1), (Y * yc).sum(1), (yb * yc).sum(1)], 1) / N
        run["triangles"] = dict(t1=float(tg[k1]), t2=float(tg[k2]), Q_ab=tri[:, 0].tolist(), Q_ac=tri[:, 1].tolist(), Q_bc=tri[:, 2].tolist(),
                                expected=[float(r["q"][:, k2].mean()), float(r["q"][:, k1].mean()), float(r["q"][:, k1].mean())])
        np.savez_compressed(ROOT / "results/E0/E0_paths_beta0.npz", t=tg, q=r["q"][:60], q_truth=r["q_truth"][:60])
        # energy-information identity
        tgE = np.r_[np.geomspace(1e-5, 0.02, 30), np.linspace(0.025, 0.6, 24)]
        en = P.energy_information_check(S, B, Y[:200], tgE, rng); run["energy"] = en
    res["runs"][str(beta)] = run
    print("beta", beta, "commit medians", {k: round(v["median"], 4) for k, v in commit.items()}, f"{time.time() - t0:.0f}s", flush=True)
res["seconds"] = time.time() - t0
(ROOT / "results/E0/E0.json").write_text(json.dumps(res, indent=1))
e = res["runs"]["0.0"]["energy"]; print("energy vs prediction (last):", e["energy"][-1], e["prediction"][-1])
print("branching:", {k: (round(v["mean_Q"], 3), round(v["mean_q_ts"], 3)) for k, v in res["runs"]["0.0"]["branching"].items()})
print("levels self-overlap", res["levels_selfoverlap"])
