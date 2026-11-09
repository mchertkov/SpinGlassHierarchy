"""Stage 6A (calibration use): the contrast identity as a reference-free posterior-calibration test.

For the TRUE filter, along planted paths,
    E[ H(C) - H(C | H_gamma) ]  =  int_0^gamma  1/2 E[ sum_c p_c ||m_c - m_parent||^2 ] dgamma'.      (*)
Both sides can be evaluated with ANY candidate posterior q (a sampler, a variational
approximation, a learned network) using only planted paths -- no exact reference.
If q is not the Bayes filter the martingale property fails and (*) is violated.
We test this with exact N=16 enumeration and deliberately mis-specified posteriors:
  (a) wrong coupling temperature beta' in place of beta_J = 2.5,
  (b) wrong signal-to-noise clock (posterior uses lambda*H instead of H).
For each we report the identity gap and the true error of q's class information.
"""
from __future__ import annotations
import time
import numpy as np
from e6_common import ExactHierarchy, level_quantities, brownian_fields, entropy, OUT, json_dump, N16_ACCEPTED, BETA_J
from e6a_contrast_algorithms import cumtrapz0, GRID


def curves(E, H, lm32, scale=1.):
    K, n, N = H.shape
    E._lm32_backup = E._lm32 if hasattr(E, '_lm32') else None
    E.moments(H[0, :2])  # build design matrices
    E._lm32 = lm32.astype(np.float32)
    dC = np.empty((K, n)); dF = np.empty((K, n)); rc = np.empty((K, n)); rf = np.empty((K, n))
    for k in range(K):
        leaf, mom, m = E.moments(scale * H[k])
        Hc, Hf, a, b = level_quantities(leaf, mom, m)
        dC[k], dF[k] = Hc, Hf; rc[k], rf[k] = a, b
    return dC, dF, rc, rf


def run(seed, npaths=1536):
    t0 = time.monotonic()
    E = ExactHierarchy('bethe', seed)
    rng = np.random.default_rng(99100 + seed)
    Y = E.sample_prior(npaths, rng)
    H = brownian_fields(Y, GRID, rng)
    E.moments(H[0, :2])
    true_lm = E.S.logmu.copy()
    energy = (true_lm + E.S.logZ0) / BETA_J          # = H_J(y) up to a constant
    out = []
    # reference: exact filter
    variants = [('exact', BETA_J, 1.)] + [('beta', b, 1.) for b in (1.75, 2.25, 2.75, 3.25)] + [('snr', BETA_J, l) for l in (.8, 1.25)]
    exact_I = None
    for kind, b, lam in variants:
        lw = b * energy; lw = lw - np.logaddexp.reduce(lw)
        # q's prior class masses (needed for its own entropy drop)
        qmu = np.exp(lw); qpi = qmu @ E.member; qpc = qpi.reshape(2, 2).sum(1)
        Hc, Hf, rc, rf = curves(E, H, lw, lam)
        drop = np.stack([entropy(qpc) - Hc.mean(1), (entropy(qpi) - entropy(qpc)) - Hf.mean(1)], 1)
        # chain rule for the clock-scaled filter: q's information rate is lambda * contrast (d(lambda gamma)/dgamma)
        Icon = cumtrapz0(np.stack([rc.mean(1), rf.mean(1)], 1) * lam, GRID)
        if kind == 'exact': exact_I = .5 * (drop + Icon)
        k5 = len(GRID) - 1
        k32 = np.searchsorted(GRID, 3.2 - 1e-12)
        gap = drop - Icon
        # per-path martingale residual  M = [H_q(C)-H_q(C|H)] - lambda*int R^q ; z = mean/se (gamma>0)
        Mc = (entropy(qpc) - Hc) - lam * cumtrapz0(rc, GRID); Mf = ((entropy(qpi) - entropy(qpc)) - Hf) - lam * cumtrapz0(rf, GRID)
        zs = [float(np.max(np.abs(M.mean(1)[1:]) / np.maximum(M.std(1)[1:] / np.sqrt(npaths), 1e-12))) for M in (Mc, Mf)]
        row = dict(kind=kind, beta_q=b, snr_scale=lam,
                   max_abs_identity_gap=np.abs(gap).max(0).tolist(),
                   identity_gap_at_3p2=gap[k32].tolist(), max_abs_z=zs,
                   max_abs_error_vs_exact_information=np.abs(drop - exact_I).max(0).tolist(),
                   q_prior_leaf_mass=qpi.tolist())
        out.append(row)
        print(seed, kind, b, lam, 'gap', np.round(row['max_abs_identity_gap'], 3), 'err', np.round(row['max_abs_error_vs_exact_information'], 3), 'z', np.round(zs, 1), flush=True)
    res = dict(seed=seed, npaths=npaths, grid=GRID, rows=out, seconds=round(time.monotonic() - t0, 1),
               note='Exact enumeration; q differs from the Bayes filter only through beta or an SNR rescaling of the field.')
    json_dump(OUT / 'E6A' / f'calibration_seed{seed}.json', res)
    return res


if __name__ == '__main__':
    for s in N16_ACCEPTED: run(s)
