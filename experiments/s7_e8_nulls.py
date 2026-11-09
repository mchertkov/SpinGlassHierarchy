"""E8 (proxy for step 1) -- do the static predictor and dynamic detectors also 'work' without a hierarchy?
star  : 8 random orthogonal patterns + mirrors (16 states, no tree); coarse = arbitrary split of patterns into 2 groups
twost : one pattern + mirror; 'fine' = arbitrary random-projection split inside each state (CLT class posterior)
Usage: python3 experiments/e8_nulls.py star|twost   -> results/E7_null_<tag>.json (same format as E7)"""
import sys, time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "experiments"))
import pidlib_lite as P
from s7_e7_static_dynamic import info_curves, summarise

N, m0, n = 500, 0.9, 300; rng = np.random.default_rng(31); gam = np.geomspace(1e-5, 30, 46)
tag = sys.argv[1]; t0 = time.time()
if tag == "star":
    pats = np.where(rng.random((8, N)) < 0.5, 1.0, -1.0); m = m0 * np.r_[pats, -pats]; W = np.ones(16) / 16
    lab = np.array([[s, k // 4, k] for s in (0, 1) for k in range(8)])                  # sign, arbitrary group, pattern
else:
    pat = np.where(rng.random(N) < 0.5, 1.0, -1.0); m = m0 * np.stack([pat, -pat]); W = np.array([0.5, 0.5]); lab = np.array([[0], [1]])
    r = rng.standard_normal(N)                                                            # random projection for the 'fine' split
S = P.ProductMixtureSolver(m, W, lab)
a = rng.choice(S.A, n, p=S.W); Y = np.where(rng.random((n, N)) < (1 + S.m[a]) / 2, 1.0, -1.0)
Wb = np.cumsum(rng.standard_normal((len(gam), n, N)) * np.sqrt(np.diff(np.r_[0, gam]))[:, None, None], 0)
chi = np.zeros((n, len(gam))); q = np.zeros((n, len(gam)))
if tag == "star":
    pt = {l: np.zeros((n, len(gam))) for l in range(3)}; keys = [lab[:, : l + 1] for l in range(3)]
    for k, g in enumerate(gam):
        H = g * Y + Wb[k]; post = S.post_states(H); chi[:, k] = S.chi_sg(H); q[:, k] = (S.yhat(H) ** 2).sum(1) / N
        for l in range(3):
            same = (keys[l][None, :, :] == keys[l][a][:, None, :]).all(-1); pt[l][:, k] = (post * same).sum(1)
    prior = {l: np.array([S.W[(keys[l] == keys[l][aa]).all(1)].sum() for aa in a]) for l in range(3)}
    I = {l: info_curves(pt[l], prior[l], gam) for l in range(3)}; Hl = {l: float(-np.mean(np.log(prior[l]))) for l in range(3)}
    levels = {"sign": (I[0], Hl[0]), "arbitrary group|sign": (I[1] - I[0], Hl[1] - Hl[0]), "pattern|group": (I[2] - I[1], Hl[2] - Hl[1])}
else:
    side_true = (Y @ r > 0).astype(int); ps, pf = np.zeros((n, len(gam))), np.zeros((n, len(gam)))
    for k, g in enumerate(gam):
        H = g * Y + Wb[k]; logz, mh = S._tilt(H); post = np.exp(logz - logz.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        chi[:, k] = S.chi_sg(H); q[:, k] = (S.yhat(H) ** 2).sum(1) / N
        mu_r = np.einsum("ban,n->ba", mh, r); sd_r = np.sqrt(np.einsum("ban,n->ba", 1 - mh ** 2, r ** 2))
        p_pos = ndtr(mu_r / sd_r)                                                          # P(r.y > 0 | state, h)
        ps[:, k] = post[np.arange(n), a]
        ps_side = np.where(side_true[:, None] == 1, p_pos, 1 - p_pos)                      # (n, A)
        pf[:, k] = post[np.arange(n), a] * ps_side[np.arange(n), a]
    p0 = ndtr(np.einsum("an,n->a", m, r) / np.sqrt(np.einsum("an,n->a", 1 - m ** 2, r ** 2)))
    prior_f = 0.5 * np.where(side_true == 1, p0[a], 1 - p0[a])
    Ic = info_curves(ps, np.full(n, 0.5), gam); If = info_curves(pf, prior_f, gam)
    Hc = float(np.log(2)); Hf = float(-np.mean(np.log(prior_f)))
    levels = {"sign": (Ic, Hc), "arbitrary projection|sign": (If - Ic, Hf - Hc)}
C0 = (S.W[:, None, None] * np.einsum("an,am->anm", S.m, S.m)).sum(0); C0[np.diag_indices(N)] = 1.0
summarise(f"null_{tag}", gam, chi, q, levels, None, C0, dict(T4={}))
print(f"{time.time() - t0:.0f}s")
