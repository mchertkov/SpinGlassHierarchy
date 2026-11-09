"""E9 (step 2) -- static predictors and dynamic detectors at N = 32-64 via parallel tempering + pool reweighting.
Usage: python3 experiments/e9_larger.py bethe|planar <N or L> <seed>   -> results/E9_<tag>.json"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "experiments"))
import pidlib_lite as P
from s7_e7_static_dynamic import crossing_log, gaussian_surrogate

ext, size, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); t0 = time.time(); bJ = 2.5
inst = P.bethe_spin_glass(size, 3, bJ, "gauss", seed) if ext == "bethe" else P.planar_ea(size, bJ, "gauss", seed)
N = inst.N; rng = np.random.default_rng(1000 + seed)
betas = np.r_[np.geomspace(0.2, 1.0, 6)[:-1], np.linspace(1.0, bJ, 10)]
smp, acc = P.pt_sample(inst, betas, 120, 4000, 1000, 10, rng)
C = smp.shape[1]; cut = int(0.8 * C); pool = smp[:, :cut].reshape(-1, N).astype(np.float32); tr = smp[:, cut:].reshape(-1, N).astype(float)
half = len(smp) // 2; e1 = inst.energy(smp[:half].reshape(-1, N)).mean() / N; e2 = inst.energy(smp[half:].reshape(-1, N)).mean() / N
# ---- static: C0, <Q^2>, spectrum
C0 = np.cov(pool.T.astype(float), bias=True) + np.outer(pool.mean(0), pool.mean(0)) * 0     # m ~ 0 in zero field
C0 = (pool.T.astype(float) @ pool.astype(float)) / len(pool)
lam = np.sort(np.linalg.eigvalsh(C0))[::-1]; chi0 = float(np.trace(C0 @ C0) / N)
# ---- classes: Z2 sector by lowest-energy pool state; fine = basins of frequent zero-T quench minima
Xp = pool.astype(float); E = inst.energy(Xp); g = Xp[np.argmax(E)]
def quench(Y):
    Y = Y.copy()
    for _ in range(4 * N):
        f = Y @ inst.Jdense; bad = Y * f < 0
        if not bad.any(): break
        i = np.argmax(np.where(bad, -Y * f, -np.inf), 1); rows = np.where(bad.any(1))[0]; Y[rows, i[rows]] *= -1
    return Y
sub = Xp[rng.choice(len(Xp), 3000, replace=False)]; Qm = quench(sub); s = np.sign(Qm @ g); s[s == 0] = 1; Qm *= s[:, None]
uniq, cnt = np.unique(Qm.astype(np.int8), axis=0, return_counts=True); order = np.argsort(-cnt); med = []
for i in order:
    if all(abs(uniq[i].astype(float) @ uniq[j].astype(float)) / N < 0.75 for j in med): med.append(i)
    if len(med) == 3: break
Mv = uniq[med].astype(float); K = len(med)
def labels(Y):
    sg = np.sign(Y @ g); sg[sg == 0] = 1; return (sg > 0).astype(int), (sg > 0).astype(int) * K + np.argmax((Y * sg[:, None]) @ Mv.T, 1)
cp, fp = labels(Xp); ct_all, ft_all = labels(tr)
pic = np.bincount(cp, minlength=2) / len(cp); pif = np.bincount(fp, minlength=2 * K) / len(fp)
# ---- dynamics on planted paths
n = 200; idx = rng.choice(len(tr), n, replace=False); Y = tr[idx]; ct, ft = ct_all[idx], ft_all[idx]
gam = np.geomspace(1e-4, 3.0, 28)
W = np.cumsum(rng.standard_normal((len(gam), n, N)) * np.sqrt(np.diff(np.r_[0, gam]))[:, None, None], 0)
Ic, If = np.eye(2)[cp].astype(np.float32), np.eye(2 * K)[fp].astype(np.float32)
chi = np.full((n, len(gam)), np.nan); ess = np.zeros(len(gam)); pc = np.zeros((n, len(gam))); pf = np.zeros((n, len(gam)))
for k, gg in enumerate(gam):
    H = (gg * Y + W[k]).astype(np.float32); lw = H @ pool.T; w = np.exp(lw - lw.max(1, keepdims=True)).astype(np.float64); w /= w.sum(1, keepdims=True)
    ess[k] = np.median(1 / (w ** 2).sum(1))
    pc[:, k] = (w @ Ic)[np.arange(n), ct]; pf[:, k] = (w @ If)[np.arange(n), ft]
    for b in range(n):
        Xs = Xp[rng.choice(len(Xp), 2000, p=w[b])]                              # importance resampling (unbiased)
        mm = Xs.mean(0); Cm = Xs.T @ Xs / len(Xs) - np.outer(mm, mm); chi[b, k] = (Cm ** 2).sum() / N
valid = ess >= 200
lg = lambda p: np.log(np.maximum(p, 1e-12))
Ico = lg(pc).mean(0) - np.mean(np.log(pic[ct])); Ifi = lg(pf).mean(0) - np.mean(np.log(pif[ft]))
Hc = float(-np.mean(np.log(pic[ct]))); Hf = float(-np.mean(np.log(pif[ft])))
fc, ff = Ico / Hc, (Ifi - Ico) / (Hf - Hc)
gv = gam[valid]
win = dict(coarse=dict(g10=crossing_log(gv, fc[valid], .1), half=crossing_log(gv, fc[valid], .5), g90=crossing_log(gv, fc[valid], .9)),
           fine=dict(g10=crossing_log(gv, ff[valid], .1), half=crossing_log(gv, ff[valid], .5), g90=crossing_log(gv, ff[valid], .9)))
r = gam[None] * chi; chiG = gaussian_surrogate(C0, gam)[1]
det = dict(lograte=r.mean(0), var_chi=chi.var(0), std_lograte=r.std(0), excess=chi.mean(0) - chiG)
peaks = {k: float(gv[int(np.nanargmax(v[valid]))]) for k, v in det.items()}
res = dict(tag=f"{ext}_{N}_s{seed}", N=N, swap_acc=acc.tolist(), energy_halves=[float(e1), float(e2)], pool=int(len(pool)), K=K,
           chi0=chi0, Q2=chi0 / N, chi_first=float(np.nanmean(chi[:, 0])), lam=lam[:10].tolist(), lam_over_N=(lam[:6] / N).tolist(),
           spectral=(1 / lam[:4]).tolist(), gamma=gam.tolist(), ess=ess.tolist(), gamma_valid_max=float(gv[-1]),
           windows=win, frac_coarse=fc.tolist(), frac_fine=ff.tolist(), detectors={k: v.tolist() for k, v in det.items()}, peaks=peaks, seconds=time.time() - t0)
(ROOT / f"results/S7/E9_{res['tag']}.json").write_text(json.dumps(res, default=float))
print(res["tag"], f"E halves {e1:.4f}/{e2:.4f} chi0={chi0:.2f} chi(g0)={res['chi_first']:.2f} lam/N={np.round(lam[:5] / N, 3).tolist()}",
      f"valid<= {gv[-1]:.3g} K={K}", "windows", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in win.items()},
      "1/lam", np.round(1 / lam[:3], 4).tolist(), "peaks", {k: round(v, 4) for k, v in peaks.items()}, f"{time.time() - t0:.0f}s")
