"""N2-B (2026-10-31): how many decisions does one trajectory make?

For every planted trajectory (exact posteriors, N=16 and 20) we record
  * the commitment time (planted-class posterior first >= 0.9) at every level that SPLITS on the trajectory's own branch,
  * recrossings (later < 0.8),
  * label-free traces: the per-path localization rate r(gamma) = gamma Tr(C_gamma^2)/N and the leading eigenvalue of C_gamma.
Distinct decision events are commitments separated by more than a factor f in gamma (f = 1.5 and 2).
usage: python experiments/n2_paths.py --N 16 --betas 1.0 1.5 2.0 2.5 --seeds 1-30 --paths 128
"""
from pathlib import Path
import sys, json, time, argparse
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'experiments')]
import exacttree as E
from n1_levels import J_of, parse_seeds
OUT = ROOT / 'results' / 'N2'; OUT.mkdir(parents=True, exist_ok=True)


def events(c, f):
    """number of distinct events among increasing commitment times c (nan-free), merging ratios < f"""
    if len(c) == 0: return 0
    n = 1
    for a, b in zip(c[:-1], c[1:]):
        if b / a >= f: n += 1
    return n


def analyze(family, N, beta, seed, npaths, ng=40):
    rng = np.random.default_rng(424242 + seed + int(100 * beta))
    M = E.ExactModel(J_of(family, N, beta, seed)); T = E.build_tree(M)
    labs, Ks = T['labels'], T['n_classes']; L = len(labs)
    g = np.concatenate([[0.0], np.geomspace(0.02 / N, 5.0, ng)])
    Y, w = E.stratified_endpoints(M, labs[-1], Ks[-1], npaths, rng); n = len(Y)
    dg = np.diff(np.concatenate([[0.0], g]))
    H = g[None, :, None] * Y[:, None, :] + np.cumsum(rng.standard_normal((n, len(g), N)) * np.sqrt(dg)[None, :, None], 1)
    a = ((Y[:, :M.na] < 0) @ (1 << np.arange(M.na))).astype(int); b = ((Y[:, M.na:] < 0) @ (1 << np.arange(M.nb))).astype(int)
    plant = [lab[a, b] for lab in labs]
    # does level l split the planted branch? (parent at l-1 has two children)
    splits = np.zeros((n, L), bool); splits[:, 0] = True
    for l in range(1, L):
        for i in range(n):
            splits[i, l] = len(np.unique(labs[l][labs[l - 1] == plant[l - 1][i]])) > 1
    pp = np.zeros((n, len(g), L)); r = np.zeros((n, len(g))); lmax = np.zeros((n, len(g)))
    for i in range(n):
        for k in range(len(g)):
            W = M.posterior(H[i, k]) if g[k] > 0 else M.p
            for l in range(L): pp[i, k, l] = W[labs[l] == plant[l][i]].sum()
            m = M.first_moment(W); C = M.second_moment(W) - np.outer(m, m)
            r[i, k] = g[k] * (C * C).sum() / N; lmax[i, k] = np.linalg.eigvalsh(C)[-1]
    commit = np.full((n, L), np.nan); recross = np.zeros((n, L), bool)
    for i in range(n):
        for l in range(L):
            if not splits[i, l]: continue
            h = np.flatnonzero(pp[i, :, l] >= 0.9)
            if len(h):
                commit[i, l] = g[h[0]]; recross[i, l] = bool((pp[i, h[0]:, l] < 0.8).any())
    nlev = splits.sum(1)
    ev15 = np.array([events(np.sort(commit[i][~np.isnan(commit[i])]), 1.5) for i in range(n)])
    ev20 = np.array([events(np.sort(commit[i][~np.isnan(commit[i])]), 2.0) for i in range(n)])
    out = dict(family=family, N=N, beta=beta, seed=seed, n_levels_glassy=L - 1, npaths=int(n), w=w.tolist(),
               levels_on_branch=nlev.tolist(), events_f15=ev15.tolist(), events_f20=ev20.tolist(),
               commit=np.where(np.isnan(commit), -1, commit).tolist(), recross=recross.tolist(),
               kappa=[[c['kappa'] for c in ch] for ch in E.parent_channels(M, T)])
    np.savez_compressed(OUT / f'paths_{family}_N{N}_b{beta}_s{seed}.npz', gamma=g, w=w, r=r, lmax=lmax, pp=pp, splits=splits, commit=commit)
    json.dump(out, open(OUT / f'paths_{family}_N{N}_b{beta}_s{seed}.json', 'w'))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', default='bethe'); ap.add_argument('--N', type=int, default=16)
    ap.add_argument('--betas', type=float, nargs='*', default=[2.5]); ap.add_argument('--seeds', default='1-30')
    ap.add_argument('--paths', type=int, default=128)
    a = ap.parse_args()
    for beta in a.betas:
        for s in parse_seeds(a.seeds):
            if (OUT / f'paths_{a.family}_N{a.N}_b{beta}_s{s}.json').exists(): continue
            tic = time.time(); o = analyze(a.family, a.N, beta, s, a.paths)
            w = np.array(o['w'])
            print(a.family, a.N, beta, s, 'glassy', o['n_levels_glassy'], 'mean levels/path %.2f' % (w @ np.array(o['levels_on_branch'])),
                  'events f2 %.2f' % (w @ np.array(o['events_f20'])), '%.0fs' % (time.time() - tic), flush=True)
