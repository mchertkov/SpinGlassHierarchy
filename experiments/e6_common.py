"""Stage 6 shared utilities: frozen hierarchies, equilibrium libraries, exact N=16 objects.

Stage 6 adds (i) algorithmic uses of the class-contrast (entropy-production) identity and
(ii) a learned-denoiser probe of whether a trained generator's hidden units represent the
frozen hierarchy.  Nothing here refits any Stage-5 hierarchy: all trees are *replayed* from
the frozen Stage-5 seeds/data and checked against the archived JSON values.
"""
from __future__ import annotations
from pathlib import Path
import sys, json, time
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'experiments')]
import pidlib as P
from e3b_e1_heldout_hierarchy import train_tree, predict, adjusted_rand

OUT = ROOT / 'results' / 'E6'
OUT.mkdir(parents=True, exist_ok=True)
BETA_J = 2.5
N16_ACCEPTED = (1, 3, 4, 5, 8, 9, 11)
N16_REJECTED = (2, 6, 7, 10, 12)
LARGE_DISORDERS = (4, 5, 8)
M0 = .8  # two-state null magnetisation amplitude (Stage 5/5E value)


def entropy(p, axis=-1):
    p = np.asarray(p, float)
    return -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.), axis=axis)


# ----------------------------------------------------------------------------- N=16 exact
def n16_solver_and_tree(model, seed):
    """Replay the Stage-5 exact N=16 hierarchy (e5_exact_transients.analyze), bit-for-bit."""
    from e5_exact_transients import setup
    rng = np.random.default_rng(202611190 + seed * 101 + {'bethe': 0, 'two_state': 1, 'product': 2}[model] * 1000000)
    S = setup(model, seed)
    t1 = S.sample_prior(12000, rng); t2 = S.sample_prior(12000, rng)
    treeA = train_tree(np.r_[t1, -t1]); treeB = train_tree(np.r_[t2, -t2])
    held = S.sample_prior(5000, rng)
    ca, fa = predict(held, treeA); cb, fb = predict(held, treeB)
    aris = [adjusted_rand(ca, cb), adjusted_rand(fa, fb)]
    return S, treeA, aris


class ExactHierarchy:
    """Exact N=16 Gibbs law + frozen tree: posterior class masses and class magnetisations."""

    def __init__(self, model, seed):
        self.model, self.seed = model, seed
        self.S, self.tree, self.aris = n16_solver_and_tree(model, seed)
        self.X = self.S.X.astype(float)
        self.N = self.X.shape[1]
        mu = np.exp(self.S.logmu); self.mu = mu / mu.sum()
        _, self.leafX = predict(self.X, self.tree)
        self.member = (self.leafX[:, None] == np.arange(4)[None, :]).astype(float)
        self.pi = self.mu @ self.member
        self.leaf_means = np.stack([(self.mu * self.member[:, l]) @ self.X / max(self.pi[l], 1e-300) for l in range(4)])

    def posterior(self, H):
        lw = H @ self.X.T + self.S.logmu[None, :]
        return np.exp(lw - logsumexp(lw, axis=1, keepdims=True))

    def moments(self, H, chunk=96):
        """Return leaf masses (B,4), leaf first moments (B,4,N) [= p_l m_l], and m (B,N)."""
        if not hasattr(self, '_D'):
            N = self.N
            self._D = np.concatenate([self.X, self.member,
                                      (self.member[:, :, None] * self.X[:, None, :]).reshape(len(self.X), 4 * N)], 1).astype(np.float32)
            self._X32 = self.X.astype(np.float32); self._lm32 = self.S.logmu.astype(np.float32)
        N = self.N; outs = []
        for b in range(0, len(H), chunk):
            lw = H[b:b + chunk].astype(np.float32) @ self._X32.T + self._lm32[None, :]
            lw -= lw.max(1, keepdims=True)
            w = np.exp(lw); w /= w.sum(1, keepdims=True)
            z = (w @ self._D).astype(np.float64)
            outs.append((z[:, N:N + 4], z[:, N + 4:].reshape(-1, 4, N), z[:, :N]))
        return [np.concatenate(z) for z in zip(*outs)]

    def covariance(self, h):
        w = self.posterior(h[None])[0]
        m = w @ self.X
        return (self.X.T * w) @ self.X - np.outer(m, m), w

    def sample_prior(self, n, rng):
        return self.S.sample_prior(n, rng)


def level_quantities(leaf, mom, m):
    """Per-path level entropies and contrast rates from leaf masses and first moments.

    rate_coarse = 1/2 sum_parents p_P ||m_P - m||^2 ;
    rate_fine   = 1/2 sum_leaves p_l ||m_l - m_P(l)||^2 (conditional fine information rate).
    Both are the class-contrast (entropy-production) right-hand sides of Eq. (classentropy).
    """
    par = leaf.reshape(-1, 2, 2).sum(-1)
    pmom = mom.reshape(-1, 2, 2, mom.shape[-1]).sum(2)
    E1 = (np.sum(pmom ** 2, -1) / np.maximum(par, 1e-300)).sum(-1)
    E2 = (np.sum(mom ** 2, -1) / np.maximum(leaf, 1e-300)).sum(-1)
    rc = .5 * (E1 - np.sum(m * m, -1))
    rf = .5 * (E2 - E1)
    Hc = entropy(par); Hf = entropy(leaf) - Hc
    return Hc, Hf, rc, rf


def brownian_fields(Y, grid, rng):
    H = np.empty((len(grid),) + Y.shape); h = np.zeros(Y.shape); last = 0.
    for k, g in enumerate(grid):
        d = g - last
        if d > 0: h = h + d * Y + np.sqrt(d) * rng.standard_normal(Y.shape)
        H[k] = h; last = g
    return H


# ----------------------------------------------------------------------------- N=32/64 frozen trees
def frozen_large_tree(N, disorder):
    """Stage 5B/5D tree: train_tree on Z2-symmetrised repeat-0 chains 0..7 (e5b run_large)."""
    from e5_largeN_importance import load
    a, _ = load(N, disorder, 0)
    xa = a[:, :8].reshape(-1, N)
    return train_tree(np.r_[xa, -xa])


def frozen_null_tree(model, N, seed):
    """Stage 5E matched-null tree (e5e_matched_nulls.summarize replay)."""
    from e5e_matched_nulls import sample_prior
    r = np.random.default_rng(975317 + N * 10091 + seed * 1007 + ({'product': 11, 'two_state': 91}[model]))
    trainA = sample_prior(model, 12000, N, r)
    return train_tree(np.concatenate([trainA, -trainA]))


def null_sample(model, n, N, rng):
    from e5e_matched_nulls import sample_prior
    return sample_prior(model, n, N, rng).astype(float)


def library_path(N, disorder):
    return OUT / 'libraries' / f'bethe_N{N}_disorder{disorder}.npz'


def load_library(N, disorder):
    z = np.load(library_path(N, disorder))
    return z['Y'].astype(float), z


def hierarchy_geometry(Y, tree):
    """Leaf masses, leaf means and contrast vectors from equilibrium samples."""
    _, leaf = predict(Y, tree)
    pi = np.bincount(leaf, minlength=4) / len(leaf)
    means = np.stack([Y[leaf == l].mean(0) if np.any(leaf == l) else np.zeros(Y.shape[1]) for l in range(4)])
    par_pi = pi.reshape(2, 2).sum(1)
    par_means = np.stack([(pi[2 * p] * means[2 * p] + pi[2 * p + 1] * means[2 * p + 1]) / max(par_pi[p], 1e-300) for p in range(2)])
    mbar = par_pi @ par_means
    d_coarse = par_means[0] - par_means[1]
    d_fine = np.stack([means[2 * p] - means[2 * p + 1] for p in range(2)])
    return dict(pi=pi, means=means, par_pi=par_pi, par_means=par_means, mbar=mbar,
                d_coarse=d_coarse, d_fine=d_fine, leaf=leaf)


def json_dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
