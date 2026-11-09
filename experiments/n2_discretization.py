"""N2-A (2026-10-31): does the PID protocol nu(t) matter for a generator that takes finitely many steps?

The ideal generator's decisions are protocol-independent on the internal clock gamma (Proposition 1). A practical generator
integrates dX = [kappa(t) yhat(kappa X) - c(t) X] dt + dW with K Euler steps, uniformly spaced in physical time t, and decodes
y = sign(X_1). Here the drift is EXACT (enumeration), so every error comes from time discretization and from where the
protocol places the decisions on the t axis.

Protocols (harmonic PID, constant stiffness nu = S^2):
  'follmer' : S = 0                      gamma = t/(1-t): decisions spread over roughly the first 40% of t, half of t in the local stage
  'tuned'   : S from the binary-channel law, so that the predicted decision time gamma* is reached at t = 1/2
  'stiff'   : S = 12                     decisions late, little time left for local resolution
Endpoint error: total variation between the class frequencies of the decoded endpoints and the exact class masses, per level,
minus the sampling-noise floor of the same number of exact samples.

usage: python experiments/n2_discretization.py [--seeds-per-beta 6] [--M 3000]
"""
from pathlib import Path
import sys, json, time, argparse
import numpy as np
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'experiments')]
import exacttree as E
from n1_levels import pred_half
OUT = ROOT / 'results' / 'N2'; OUT.mkdir(parents=True, exist_ok=True)
S10, S90 = 0.1488, 3.759


def coeffs(S, t):
    if S == 0: return 1 / (1 - t), 1 / (1 - t)
    return S / np.sinh(S * (1 - t)), S / np.tanh(S * (1 - t))


def gamma_of(S, t):
    t = np.asarray(t, float)
    if S == 0: return t / (1 - t)
    return S * np.sinh(S * t) / (np.sinh(S) * np.sinh(S * (1 - t)))


def tuned_S(gstar):
    """S such that gamma_S(1/2) = S/sinh(S) = gstar (requires gstar < 1)."""
    return brentq(lambda S: S / np.sinh(S) - gstar, 1e-6, 60)


def run_instance(beta, seed, Ks=(2, 4, 8, 16, 32, 64), M=4000, rng=None):
    N = 16
    Mod = E.ExactModel(E.bethe_J(N, beta, seed)); T = E.build_tree(Mod); ch = E.parent_channels(Mod, T)
    L = len(T['labels'])
    # predicted decision times per level and the protocol targets
    ph = [pred_half(c) for c in ch]
    gstar = float(np.exp(np.mean(np.log(ph[:2]))))
    protos = {'follmer': 0.0, 'tuned': tuned_S(gstar), 'stiff': 12.0}
    # truncated configuration set (discarded prior mass < 1e-6)
    lp = Mod.logp.ravel(); keep = np.flatnonzero(lp > lp.max() - 25); Y = Mod.config(keep).astype(np.float64)
    lpk = lp[keep]
    labs = [lab.ravel() for lab in T['labels']]; Ks_cls = T['n_classes']
    pis = [np.bincount(lab, weights=Mod.p.ravel(), minlength=K) for lab, K in zip(labs, Ks_cls)]
    full_index = lambda y: (((y[:, :Mod.na] < 0) @ (1 << np.arange(Mod.na))) * 2 ** Mod.nb
                            + ((y[:, Mod.na:] < 0) @ (1 << np.arange(Mod.nb)))).astype(int)

    def yhat(H):
        out = np.empty_like(H)
        for s in range(0, len(H), 500):
            z = lpk[None, :] + H[s:s + 500] @ Y.T; z -= z.max(1, keepdims=True); w = np.exp(z); w /= w.sum(1, keepdims=True)
            out[s:s + 500] = w @ Y
        return out

    def tv(idx):
        return [0.5 * float(np.abs(np.bincount(lab[idx], minlength=K) / len(idx) - pi).sum()) for lab, K, pi in zip(labs, Ks_cls, pis)]

    Elog = float((Mod.p.ravel() * lp).sum()); Slog = float(np.sqrt((Mod.p.ravel() * (lp - Elog) ** 2).sum()))
    # sampling-noise floor
    floor = np.mean([tv(np.searchsorted(np.cumsum(Mod.p.ravel()), rng.random(M) * Mod.p.sum())) for _ in range(20)], axis=0).tolist()
    res = {}
    for name, S in protos.items():
        res[name] = {}
        for K in Ks:
            X = np.zeros((M, N)); dt = 1.0 / K
            for k in range(K):
                t = k * dt; kap, c = coeffs(S, t)
                X = X + (kap * yhat(kap * X) - c * X) * dt + np.sqrt(dt) * rng.standard_normal((M, N))
            y = np.where(X >= 0, 1.0, -1.0)
            # steps whose gamma lies in each level's predicted window (union over parents)
            tk = np.arange(K) * dt; gk = gamma_of(S, tk)
            win = []
            for c_ in ch:
                lo = min(S10 / x['kappa'] for x in c_); hi = max(S90 / x['kappa'] for x in c_)
                win.append(int(((gk >= lo) & (gk <= hi)).sum()))
            fi = full_index(y)
            res[name][K] = dict(tv=tv(fi), steps_in_window=win, logmu_bias=float((lp[fi].mean() - Elog) / Slog))
    return dict(beta=beta, seed=seed, n_levels_glassy=L - 1, pred_half=ph, gstar=gstar, S={k: float(v) for k, v in protos.items()},
                n_kept=int(len(keep)), floor=floor, res=res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds-per-beta', type=int, default=6); ap.add_argument('--M', type=int, default=4000)
    ap.add_argument('--betas', type=float, nargs='*', default=[2.5, 1.5])
    a = ap.parse_args()
    rng = np.random.default_rng(31102026)
    for beta in a.betas:
        done = 0
        for seed in range(1, 61):
            f = ROOT / 'results' / 'N1S' / f'bethe_N16_b{beta}_s{seed}.json'
            if not f.exists() or json.load(open(f))['n_levels_glassy'] < 1: continue
            outf = OUT / f'disc_N16_b{beta}_s{seed}.json'
            if not outf.exists():
                tic = time.time(); o = run_instance(beta, seed, M=a.M, rng=rng); o['seconds'] = time.time() - tic
                json.dump(o, open(outf, 'w'), indent=1)
                print(beta, seed, 'S', {k: round(v, 2) for k, v in o['S'].items()}, 'floor', np.round(o['floor'], 3),
                      {p: [(round(o['res'][p][K]['tv'][-1], 3), round(o['res'][p][K]['logmu_bias'], 2)) for K in o['res'][p]] for p in o['res']}, '%.0fs' % o['seconds'], flush=True)
            done += 1
            if done >= a.seeds_per_beta: break
