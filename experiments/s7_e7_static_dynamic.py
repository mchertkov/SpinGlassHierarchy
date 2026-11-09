"""E7 -- Static predictors and dynamic detectors of decisions.
T1  initial localisation rate:  d qbar/d gamma |_0 = chi_SG(0) = N <Q^2>_J
T2  decision index  D(gamma) = Var_paths[chi_SG] / E[chi_SG]^2  (identically 0 for a Gaussian target),
    versus measured class-decision windows; Gaussian surrogate with the same equilibrium covariance
T3  spectral predictor: eigenvalues of the equilibrium correlation matrix C0, windows gamma_k ~ 1/lambda_k
T4  Jacobian route: d yhat / d h = C_t (finite differences) and Hutchinson estimate of Tr C^2
Usage: python3 experiments/e7_static_dynamic.py E0 | bethe <seed> | planar <seed>   -> results/E7_<tag>.json"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code"))
import pidlib_lite as P


def crossing_log(g, r, lam=0.5):
    for k in range(1, len(g)):
        if r[k - 1] < lam <= r[k]:
            f = (lam - r[k - 1]) / (r[k] - r[k - 1]); return float(np.exp(np.log(g[k - 1]) + f * (np.log(g[k]) - np.log(g[k - 1]))))
    return float("nan")


def gaussian_surrogate(C0, gam):
    lam = np.clip(np.linalg.eigvalsh(C0), 1e-12, None); N = len(lam)
    post = lam[None] / (1 + gam[:, None] * lam[None])                 # eigenvalues of (C0^-1 + g)^-1
    return (lam.sum() - post.sum(1)) / N, (post ** 2).sum(1) / N        # qbar_G, chi_G


def info_curves(ptrue, prior_true, gam):
    """I(C;H) = E log p_true/pi_true along gamma, for a label; ptrue (n, G)."""
    return np.mean(np.log(np.maximum(ptrue, 1e-300)), 0) - np.mean(np.log(prior_true))


def summarise(tag, gam, chi, q, info_levels, H_levels, C0, extra):
    N = C0.shape[0]; qG, chiG = gaussian_surrogate(C0, gam)
    D = chi.var(0) / chi.mean(0) ** 2
    lam = np.sort(np.linalg.eigvalsh(C0))[::-1]
    win = {}
    for nm, (I, Hc) in info_levels.items():
        win[nm] = dict(gamma_half=crossing_log(gam, I / Hc, 0.5), gamma_10=crossing_log(gam, I / Hc, 0.1),
                       gamma_90=crossing_log(gam, I / Hc, 0.9), terminal=float(Hc), frac=(I / Hc).tolist())
    kD = int(np.argmax(D)); chi0 = float(np.trace(C0 @ C0) / N)
    r = gam[None] * chi; det = dict(lograte_mean=r.mean(0), var_chi=chi.var(0), std_lograte=r.std(0), excess=chi.mean(0) - chiG)
    peaks = {k: float(gam[int(np.argmax(v))]) for k, v in det.items()}
    k_ex = np.where(det["excess"] > 0)[0]
    peaks["excess_positive_range"] = [float(gam[k_ex.min()]), float(gam[k_ex.max()])] if len(k_ex) else None
    slope0 = float((q.mean(0)[0]) / gam[0])
    res = dict(tag=tag, N=N, gamma=gam.tolist(), chi_mean=chi.mean(0).tolist(), chi_G=chiG.tolist(), q_mean=q.mean(0).tolist(), q_G=qG.tolist(),
               decision_index=D.tolist(), gamma_at_max_D=float(gam[kD]), max_D=float(D[kD]),
               chi0_exact=chi0, Q2_exact=chi0 / N, slope_qbar_at_0=slope0, chi_mean_smallest_gamma=float(chi.mean(0)[0]),
               detectors={k: v.tolist() for k, v in det.items()}, detector_peaks=peaks,
               top_eigs=lam[:8].tolist(), spectral_windows=(1 / lam[:6]).tolist(), windows=win, **extra)
    (ROOT / f"results/S7/E7_{tag}.json").write_text(json.dumps(res))
    print(tag, f"T1 chi0={chi0:.3f} slope={slope0:.3f}", "| T2 peaks", {k: (np.round(v, 4) if v is not None else None) for k, v in peaks.items()},
          "| windows", {k: (round(v['gamma_10'], 4), round(v['gamma_half'], 4), round(v['gamma_90'], 4)) for k, v in win.items()},
          "| T3 1/lambda", np.round(res['spectral_windows'][:4], 4).tolist(), "lam", np.round(lam[:5], 2).tolist(), flush=True)
    return res


def run_E0():
    S = P.ultrametric_caricature(N=500); N = S.N; lab = S.labels; rng = np.random.default_rng(7)
    n = 300; gam = np.geomspace(1e-5, 1.0, 40)
    a = rng.choice(S.A, n, p=S.W); Y = np.where(rng.random((n, N)) < (1 + S.m[a]) / 2, 1.0, -1.0)
    W = np.cumsum(rng.standard_normal((len(gam), n, N)) * np.sqrt(np.diff(np.r_[0, gam]))[:, None, None], 0)
    chi = np.zeros((n, len(gam))); q = np.zeros((n, len(gam))); pt = {l: np.zeros((n, len(gam))) for l in range(4)}
    keys = [lab[:, : l + 1] for l in range(4)]
    for k, g in enumerate(gam):
        H = g * Y + W[k]; post = S.post_states(H); yh = S.yhat(H)
        chi[:, k] = S.chi_sg(H); q[:, k] = (yh ** 2).sum(1) / N
        for l in range(4):
            same = (keys[l][None, :, :] == keys[l][a][:, None, :]).all(-1); pt[l][:, k] = (post * same).sum(1)
    prior = {l: np.array([S.W[(keys[l] == keys[l][aa]).all(1)].sum() for aa in a]) for l in range(4)}
    I = {l: info_curves(pt[l], prior[l], gam) for l in range(4)}
    Hl = {l: float(-np.mean(np.log(prior[l]))) for l in range(4)}
    levels = {"sign": (I[0], Hl[0])}
    for l in (1, 2, 3): levels[f"level{l}|{l-1}"] = (I[l] - I[l - 1], Hl[l] - Hl[l - 1])
    C0 = (S.W[:, None, None] * (np.einsum("an,am->anm", S.m, S.m))).sum(0); C0[np.diag_indices(N)] = 1.0
    # T4 on the product mixture: finite-difference Jacobian of yhat wrt h vs C (one field)
    h0 = gam[20] * Y[:1] + W[20][:1]; eps = 1e-4; J = np.zeros((N, N))
    for i in range(N):
        e = np.zeros((1, N)); e[0, i] = eps; J[:, i] = (S.yhat(h0 + e)[0] - S.yhat(h0 - e)[0]) / (2 * eps)
    trJ2 = float((J ** 2).sum() / N); chi_ex = float(S.chi_sg(h0)[0])
    vs = np.where(rng.random((16, N)) < 0.5, 1.0, -1.0); hut = float(np.mean([((S.yhat(h0 + eps * v[None])[0] - S.yhat(h0 - eps * v[None])[0]) / (2 * eps)) @ ((S.yhat(h0 + eps * v[None])[0] - S.yhat(h0 - eps * v[None])[0]) / (2 * eps)) for v in vs]) / N)
    return summarise("E0", gam, chi, q, levels, Hl, C0, dict(T4=dict(chi_exact=chi_ex, jacobian_frobenius=trJ2, hutchinson16=hut)))


def classes_for_instance(S, inst):
    """Coarse: Z2 sector sign(g.y) (g = ground state; ties by the second state). Fine: basins of the heaviest
    distinct single-flip-stable configurations (folded to the + sector, |overlap| < 0.75), nearest by overlap."""
    X = S.X.astype(float); N = inst.N; order = np.argsort(-S.logmu); g = X[order[0]]
    s = np.sign(X @ g); g2 = X[order[1]] if abs(X[order[1]] @ g) < N else X[order[2]]
    s[s == 0] = np.sign(X[s == 0] @ g2); s[s == 0] = 1
    f = inst.energy(X) * 0; Jd = inst.Jdense
    field = X @ Jd; stable = np.all(X * field >= 0, axis=1)                       # single-flip stable (max of H)
    cand = [i for i in order if stable[i] and (X[i] @ g) >= 0][:200]
    med = []
    for i in cand:
        if all(abs(X[i] @ X[j]) / N < 0.75 for j in med): med.append(i)
        if len(med) == 3: break
    Mv = X[med]; fold = X * s[:, None]
    fine_idx = np.argmax(fold @ Mv.T, 1)
    coarse = (s > 0).astype(int); fine = coarse * len(med) + fine_idx
    return coarse, fine, len(med)


def run_ising(extreme, seed):
    inst = P.bethe_spin_glass(16, 3, 2.5, "gauss", seed) if extreme == "bethe" else P.planar_ea(4, 2.5, "gauss", seed)
    S = P.ExactSolver(inst); N = inst.N; X = S.X; mu = np.exp(S.logmu)
    C0 = (X.T.astype(float) * mu) @ X.astype(float)
    coarse, fine, K = classes_for_instance(S, inst)
    Ic = np.eye(2)[coarse].astype(np.float32); If = np.eye(2 * K)[fine].astype(np.float32)
    iu = np.triu_indices(N, 1); Pp = (X[:, iu[0]] * X[:, iu[1]]).astype(np.float32)
    rng = np.random.default_rng(100 + seed); n = 300; gam = np.geomspace(1e-3, 10, 32)
    idx = rng.choice(len(mu), n, p=mu / mu.sum()); Y = X[idx].astype(float)
    W = np.cumsum(rng.standard_normal((len(gam), n, N)) * np.sqrt(np.diff(np.r_[0, gam]))[:, None, None], 0)
    chi = np.zeros((n, len(gam))); q = np.zeros((n, len(gam))); pc = np.zeros((n, len(gam))); pf = np.zeros((n, len(gam)))
    for k, g in enumerate(gam):
        H = (g * Y + W[k]).astype(np.float32)
        for c0 in range(0, n, 100):
            lp = S.logmu[None] + H[c0:c0 + 100] @ X.T; w = np.exp(lp - logsumexp(lp, 1, keepdims=True)).astype(np.float32)
            m = w @ X; E2 = w @ Pp; off = E2 - (m[:, iu[0]] * m[:, iu[1]])
            chi[c0:c0 + 100, k] = (((1 - m ** 2) ** 2).sum(1) + 2 * (off ** 2).sum(1)) / N
            q[c0:c0 + 100, k] = (m ** 2).sum(1) / N
            pc[c0:c0 + 100, k] = (w @ Ic)[np.arange(len(w)), coarse[idx[c0:c0 + 100]]]
            pf[c0:c0 + 100, k] = (w @ If)[np.arange(len(w)), fine[idx[c0:c0 + 100]]]
    pic = (mu[:, None] * Ic).sum(0); pif = (mu[:, None] * If).sum(0)
    Ico = info_curves(pc, pic[coarse[idx]], gam); Ifi = info_curves(pf, pif[fine[idx]], gam)
    Hc = float(-np.mean(np.log(pic[coarse[idx]]))); Hf = float(-np.mean(np.log(pif[fine[idx]])))
    levels = {"coarse(Z2)": (Ico, Hc), "fine|coarse": (Ifi - Ico, Hf - Hc)}
    # T4: finite-difference Jacobian of yhat wrt h vs exact C at one field
    h0 = (gam[16] * Y[:1] + W[16][:1]); eps = 1e-3
    def yh(hh): return S.yhat(hh)[0].astype(float)
    J = np.stack([(yh(h0 + eps * e[None]) - yh(h0 - eps * e[None])) / (2 * eps) for e in np.eye(N)], 1)
    w0 = S.weights(h0)[0]; m0 = w0 @ X; Cx = (X.T * w0) @ X - np.outer(m0, m0)
    vs = np.where(rng.random((16, N)) < 0.5, 1.0, -1.0)
    hut = float(np.mean([np.sum(((yh(h0 + eps * v[None]) - yh(h0 - eps * v[None])) / (2 * eps)) ** 2) for v in vs]) / N)
    extra = dict(n_fine_basins=K, T4=dict(chi_exact=float((Cx ** 2).sum() / N), jacobian_frobenius=float((J ** 2).sum() / N),
                                         max_abs_J_minus_C=float(np.abs(J - Cx).max()), hutchinson16=hut), instance=inst.name)
    return summarise(f"{extreme}_s{seed}", gam, chi, q, levels, None, C0, extra)


if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1] == "E0": run_E0()
    else: run_ising(sys.argv[1], int(sys.argv[2]))
    print(f"{time.time() - t0:.0f}s")
