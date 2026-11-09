"""Stage 6A: the class-contrast (entropy-production) identity used as an algorithm.

Three practical uses of
    d/dgamma I(C_l; H | C_{l-1}) = 1/2 E[ sum_c p_c || m_c - m_parent(c) ||^2 ]          (*)
are tested with EXACT N=16 posteriors (2^16 enumeration) on the frozen Stage-5 hierarchies.

(U1) Derivative-free, variance-reduced information estimator.  The per-path entropy drop
     H(C) - H(C|H_gamma) is a submartingale whose Doob-Meyer compensator is the integral of
     the per-path contrast rate (*).  Averaging the compensator removes the martingale noise.
     We compare RMSE of the two estimators for n paths (bootstrap) against a 1536-path reference.
(U2) Information budget.  1/2 E||Y-m||^2 (total I-MMSE rate) = coarse + fine + within-leaf rate.
(U3) Prediction from equilibrium data alone.  At gamma=0, (*) is a function of prior class
     masses and class means.  A 'centroid channel' (each leaf replaced by its mean magnetisation)
     gives predicted information curves and half-times without simulating any posterior.
     Predictions are compared with measured half-times (exact N=16 here; N=32/64 in e6a_large).
"""
from __future__ import annotations
import argparse, time
import numpy as np
from scipy.special import logsumexp
from e6_common import (ExactHierarchy, level_quantities, brownian_fields, entropy, OUT, json_dump,
                       N16_ACCEPTED, N16_REJECTED)

GRID = np.r_[0., np.geomspace(1e-3, 5., 56)]
HORIZONS = (3.2, 5.0)


def cumtrapz0(y, x):
    y = np.asarray(y, float); dx = np.diff(x).reshape((-1,) + (1,) * (y.ndim - 1))
    out = np.zeros_like(y); out[1:] = np.cumsum(.5 * (y[1:] + y[:-1]) * dx, axis=0)
    return out


def half_time(curve, grid, horizon):
    k = np.searchsorted(grid, horizon - 1e-12)
    target = .5 * curve[k]
    if curve[k] <= 0: return None
    ix = np.flatnonzero(curve[:k + 1] >= target)
    if not len(ix): return None
    j = ix[0]
    if j == 0: return float(grid[0])
    # linear interpolation in log-gamma between grid points j-1, j
    g0, g1 = grid[j - 1], grid[j]; c0, c1 = curve[j - 1], curve[j]
    if g0 <= 0: return float(g1 * (target / max(c1, 1e-300)))
    f = (target - c0) / max(c1 - c0, 1e-300)
    return float(np.exp(np.log(g0) + f * (np.log(g1) - np.log(g0))))


def centroid_channel(pi, means, grid, n=6000, seed=0):
    """Predicted coarse / fine-given-coarse information when each leaf is replaced by its
    equilibrium mean magnetisation (point-mass mixture in the same Gaussian channel).
    Uses common random numbers along Brownian paths; returns (len(grid),2) curves."""
    rng = np.random.default_rng(seed)
    L, N = means.shape
    lab = rng.choice(L, size=n, p=pi / pi.sum())
    S = means[lab]
    H = brownian_fields(S, grid, rng)
    out = np.zeros((len(grid), 2))
    pc = pi.reshape(2, 2).sum(1)
    H0c, H0 = entropy(pc), entropy(pi)
    norms = (means ** 2).sum(1)
    for k, g in enumerate(grid):
        lw = H[k] @ means.T + np.log(np.maximum(pi, 1e-300))[None] - .5 * g * norms[None]
        w = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
        par = w.reshape(n, 2, 2).sum(-1)
        Hc = entropy(par); Hf = entropy(w) - Hc
        out[k] = [H0c - Hc.mean(), (H0 - H0c) - Hf.mean()]
    return out


def contrast_norms(pi, means):
    pc = pi.reshape(2, 2).sum(1)
    pm = np.stack([(pi[2 * p] * means[2 * p] + pi[2 * p + 1] * means[2 * p + 1]) / max(pc[p], 1e-300) for p in range(2)])
    mbar = pc @ pm
    Dc = float(sum(pc[p] * np.sum((pm[p] - mbar) ** 2) for p in range(2)))
    Df = float(sum(pi[l] * np.sum((means[l] - pm[l // 2]) ** 2) for l in range(4)))
    return Dc, Df


def run_instance(model, seed, npaths=1536, chunk=96):
    t0 = time.monotonic()
    E = ExactHierarchy(model, seed)
    rng = np.random.default_rng(6060600 + 101 * seed + {'bethe': 0, 'two_state': 1, 'product': 2}[model] * 77777)
    Y = E.sample_prior(npaths, rng)
    H = brownian_fields(Y, GRID, rng)
    K = len(GRID)
    Hc = np.empty((K, npaths)); Hf = np.empty((K, npaths)); rc = np.empty((K, npaths)); rf = np.empty((K, npaths))
    tot = np.empty((K, npaths)); within = np.empty((K, npaths))
    pc0 = E.pi.reshape(2, 2).sum(1)
    for k in range(K):
        for b in range(0, npaths, chunk):
            sl = slice(b, b + chunk)
            leaf, mom, m = E.moments(H[k, sl])
            a, bb, c, d = level_quantities(leaf, mom, m)
            Hc[k, sl], Hf[k, sl], rc[k, sl], rf[k, sl] = a, bb, c, d
            # within-leaf rate = 1/2 (N - sum_l ||p_l m_l||^2 / p_l); total = 1/2 (N - ||m||^2)
            tot[k, sl] = .5 * (E.N - np.sum(m * m, -1))
            within[k, sl] = .5 * (E.N - (np.sum(mom ** 2, -1) / np.maximum(leaf, 1e-300)).sum(-1))
    dropC = entropy(pc0) - Hc
    dropF = (entropy(E.pi) - entropy(pc0)) - Hf
    # Estimators of the ensemble information curves
    I_entropy = np.stack([dropC.mean(1), dropF.mean(1)], 1)
    R = np.stack([rc.mean(1), rf.mean(1)], 1)
    I_contrast = cumtrapz0(R, GRID)
    # ---- U1: bootstrap RMSE at each gamma for n paths, both estimators, vs pooled reference
    ref = .5 * (I_entropy + I_contrast)
    brng = np.random.default_rng(424242 + seed)
    nlist = (8, 24, 72, 216)
    rmse = {}
    half_sd = {}
    for n in nlist:
        eE = []; eC = []; hE = []; hC = []
        for rep in range(300):
            ii = brng.integers(0, npaths, n)
            ie = np.stack([dropC[:, ii].mean(1), dropF[:, ii].mean(1)], 1)
            ic = cumtrapz0(np.stack([rc[:, ii].mean(1), rf[:, ii].mean(1)], 1), GRID)
            eE.append(ie - ref); eC.append(ic - ref)
            hE.append([half_time(ie[:, l], GRID, 3.2) for l in range(2)])
            hC.append([half_time(ic[:, l], GRID, 3.2) for l in range(2)])
        eE = np.array(eE); eC = np.array(eC)
        rmse[n] = dict(entropy=np.sqrt((eE ** 2).mean(0)), contrast=np.sqrt((eC ** 2).mean(0)))
        hE = np.array(hE, float); hC = np.array(hC, float)
        half_sd[n] = dict(entropy_sd_log=np.nanstd(np.log(hE), 0), contrast_sd_log=np.nanstd(np.log(hC), 0))
    # finite-difference rates from the entropy curve for the n=24 path count (illustration)
    # ---- U2: information budget
    budget = dict(total=tot.mean(1), coarse=rc.mean(1), fine=rf.mean(1), within=within.mean(1))
    budget_check = float(np.max(np.abs(budget['total'] - budget['coarse'] - budget['fine'] - budget['within'])))
    # ---- U3: prediction from equilibrium data alone
    Dc, Df = contrast_norms(E.pi, E.leaf_means)
    cc = centroid_channel(E.pi, E.leaf_means, GRID, seed=seed)
    meas = {h: [half_time(I_contrast[:, l], GRID, h) for l in range(2)] for h in HORIZONS}
    meas_entropy = {h: [half_time(I_entropy[:, l], GRID, h) for l in range(2)] for h in HORIZONS}
    pred = {h: [half_time(cc[:, l], GRID, h) for l in range(2)] for h in HORIZONS}
    # within-class overlap geometry for the record
    res = dict(model=model, seed=seed, N=E.N, npaths=npaths, grid=GRID, aris=E.aris, pi=E.pi,
               accepted=bool(model == 'bethe' and seed in N16_ACCEPTED),
               I_entropy=I_entropy, I_contrast=I_contrast, rate=R,
               identity_max_abs_gap=float(np.max(np.abs(I_entropy - I_contrast))),
               identity_gap_at_horizons={h: (I_entropy[np.searchsorted(GRID, h - 1e-12)] - I_contrast[np.searchsorted(GRID, h - 1e-12)]).tolist() for h in HORIZONS},
               mc_se_entropy_at_horizons={h: (np.stack([dropC, dropF], -1)[np.searchsorted(GRID, h - 1e-12)].std(0) / np.sqrt(npaths)).tolist() for h in HORIZONS},
               rmse={str(n): {k: v for k, v in d.items()} for n, d in rmse.items()},
               half_time_log_sd={str(n): d for n, d in half_sd.items()},
               budget=budget, budget_identity_max_abs=budget_check,
               rate0_contrast_norms=dict(D2_coarse=Dc, D2_fine=Df, R0_coarse=.5 * Dc, R0_fine=.5 * Df,
                                         R0_measured=R[0].tolist()),
               centroid_prediction=cc, half_measured=meas, half_measured_entropy=meas_entropy, half_predicted=pred,
               total_information_at_5=float(np.trapezoid(budget['total'], GRID)),
               hierarchy_information_at_5=float(I_contrast[-1].sum()),
               gibbs_entropy=float(-(E.mu * np.log(np.maximum(E.mu, 1e-300))).sum()),
               seconds=round(time.monotonic() - t0, 1))
    json_dump(OUT / 'E6A' / f'{model}_seed{seed}.json', res)
    np.savez_compressed(OUT / 'E6A' / f'{model}_seed{seed}.npz', grid=GRID, dropC=dropC.astype(np.float32), dropF=dropF.astype(np.float32),
                        rc=rc.astype(np.float32), rf=rf.astype(np.float32))
    print(model, seed, 'gap', round(res['identity_max_abs_gap'], 4), 'meas', meas[3.2], 'pred', pred[3.2], 'budget err', budget_check,
          'rmse24@half', np.round(rmse[24]['entropy'].max(0), 3), np.round(rmse[24]['contrast'].max(0), 3), 's', res['seconds'], flush=True)
    return res


if __name__ == '__main__':
    (OUT / 'E6A').mkdir(exist_ok=True, parents=True)
    p = argparse.ArgumentParser(); p.add_argument('--model', default='bethe'); p.add_argument('--seeds', type=int, nargs='*', default=list(range(1, 13)))
    p.add_argument('--paths', type=int, default=1536)
    a = p.parse_args()
    for s in a.seeds: run_instance(a.model, s, a.paths)
