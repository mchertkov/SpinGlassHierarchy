"""pidlib -- Emergence of Parisi-tree in Path Integral Diffusion (single-instance toolkit).

Conventions (see paper/main.tex, Sec. 2-3)
-------------------------------------------
* Target: Ising measure mu(y) ~ exp(beta_J * sum_{i<j} J_ij y_i y_j),  y in {+-1}^N.
* Harmonic Path Integral Diffusion (H-PID) on t in [0,1], started at x(0) = 0 (all spins undecided):
      dx = u(t,x) dt + dW,   cost  E int_0^1 [ |u|^2/2 + (beta/2) |x|^2 ] dt,   x(1) ~ mu.
  Conditioned on its endpoint y the path is the harmonic-oscillator (Mehler) bridge:
      x(t) | y ~ N( a(t) y, sigma2(t) I ).
* Clock objects (s = sqrt(beta); beta -> 0 limits in brackets):
      a(t)      = sinh(s t)/sinh(s)                         [t]
      sigma2(t) = sinh(s t) sinh(s(1-t)) / (s sinh(s))      [t(1-t)]
      kappa(t)  = a/sigma2 = s / sinh(s(1-t))               [1/(1-t)]
      gamma(t)  = a*kappa  (signal-to-noise ratio)          [t/(1-t)]
      c(t)      = s coth(s(1-t))                            [1/(1-t)]
* Ising model in a moving field: P_t(y | x) ~ mu(y) exp(h.y),  h = kappa(t) x.
  Order parameter  yhat(t;x) = E[y | x(t)=x] = d/dh log Z_J(h);  drift u = kappa*yhat - c*x;
  score  grad log p_t = -x/sigma2 + kappa*yhat.
* Overlap clock q(t) = |yhat(t)|^2 / N.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr


# =============================================================================== bridge / clock
class Bridge:
    def __init__(self, beta=0.0):
        self.beta = float(beta); self.s = np.sqrt(self.beta)

    def a(self, t):
        t = np.asarray(t, float)
        return t if self.s == 0 else np.sinh(self.s * t) / np.sinh(self.s)

    def sigma2(self, t):
        t = np.asarray(t, float)
        return t * (1 - t) if self.s == 0 else np.sinh(self.s * t) * np.sinh(self.s * (1 - t)) / (self.s * np.sinh(self.s))

    def kappa(self, t):
        t = np.asarray(t, float)
        return 1 / (1 - t) if self.s == 0 else self.s / np.sinh(self.s * (1 - t))

    def c(self, t):
        t = np.asarray(t, float)
        return 1 / (1 - t) if self.s == 0 else self.s / np.tanh(self.s * (1 - t))

    def gamma(self, t):
        return self.a(t) * self.kappa(t)

    def t_of_gamma(self, g, tol=1e-12):
        """Invert the (monotone) clock gamma(t) by bisection."""
        g = np.atleast_1d(np.asarray(g, float)); lo = np.zeros_like(g); hi = np.ones_like(g) - 1e-12
        for _ in range(200):
            mid = 0.5 * (lo + hi); big = self.gamma(mid) > g
            hi = np.where(big, mid, hi); lo = np.where(big, lo, mid)
        return 0.5 * (lo + hi)

    def bridge_paths(self, y, tgrid, rng, x_start=None, t_start=0.0, n_sub=40):
        """Simulate the bridge to fixed endpoints y (B,N) from (t_start, x_start) and record x at tgrid
        (ascending, > t_start, < 1).  Euler-Maruyama on the bridge SDE dx = (kappa y - c x)dt + dW with
        n_sub substeps between grid points (exact in law up to discretisation)."""
        B, N = y.shape; x = np.zeros((B, N)) if x_start is None else x_start.copy(); t = t_start; out = []
        for tg in tgrid:
            dt = (tg - t) / n_sub
            for _ in range(n_sub):
                x = x + (self.kappa(t) * y - self.c(t) * x) * dt + np.sqrt(dt) * rng.standard_normal(x.shape); t += dt
            out.append(x.copy()); t = tg
        return np.stack(out, 1)                                     # (B, T, N)


# =============================================================================== instances
def random_regular_graph(N, d, rng, max_tries=1000):
    """Simple random d-regular graph by the pairing model with rejection; returns edge list (E,2)."""
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(N), d); rng.shuffle(stubs); e = stubs.reshape(-1, 2)
        if np.any(e[:, 0] == e[:, 1]): continue
        k = np.sort(e, 1); u = np.unique(k[:, 0] * N + k[:, 1])
        if len(u) == len(e): return k
    raise RuntimeError("no simple graph found")


def planar_lattice_edges(L):
    """Free-boundary L x L square lattice (planar); site index i = r*L + c."""
    e = []
    for r in range(L):
        for c in range(L):
            i = r * L + c
            if c + 1 < L: e.append((i, i + 1))
            if r + 1 < L: e.append((i, i + L))
    return np.array(e)


class IsingInstance:
    """Pairwise Ising instance on a sparse graph: couplings J_e on edges, inverse temperature beta_J."""

    def __init__(self, N, edges, Je, beta_J, name=""):
        self.N, self.edges, self.Je, self.beta_J, self.name = N, np.asarray(edges), np.asarray(Je, float), float(beta_J), name
        self.Jdense = np.zeros((N, N)); self.Jdense[self.edges[:, 0], self.edges[:, 1]] = self.Je; self.Jdense += self.Jdense.T
        deg = np.bincount(self.edges.ravel(), minlength=N); D = deg.max()
        self.nbr = np.full((N, D), N); self.Jn = np.zeros((N, D)); fill = np.zeros(N, int)
        for (i, j), w in zip(self.edges, self.Je):
            self.nbr[i, fill[i]] = j; self.Jn[i, fill[i]] = w; fill[i] += 1
            self.nbr[j, fill[j]] = i; self.Jn[j, fill[j]] = w; fill[j] += 1
        self.colors = self._greedy_coloring()

    def _greedy_coloring(self):
        col = -np.ones(self.N, int)
        for i in np.argsort(-np.bincount(self.edges.ravel(), minlength=self.N)):
            used = {col[j] for j in self.nbr[i] if j < self.N and col[j] >= 0}; c = 0
            while c in used: c += 1
            col[i] = c
        return [np.where(col == c)[0] for c in range(col.max() + 1)]

    def energy(self, Y):
        """H(y) = sum_e J_e y_i y_j (so that mu ~ exp(beta_J H))."""
        Y = np.asarray(Y, float); return (self.Je * Y[..., self.edges[:, 0]] * Y[..., self.edges[:, 1]]).sum(-1)


def bethe_spin_glass(N, d=3, beta_J=1.5, couplings="pm", seed=0):
    rng = np.random.default_rng(seed); e = random_regular_graph(N, d, rng)
    Je = rng.choice([-1.0, 1.0], len(e)) if couplings == "pm" else rng.standard_normal(len(e))
    return IsingInstance(N, e, Je, beta_J, f"RRG{d}_{couplings}_N{N}_b{beta_J}_s{seed}")


def planar_ea(L, beta_J=2.0, couplings="gauss", seed=0):
    rng = np.random.default_rng(seed); e = planar_lattice_edges(L)
    Je = rng.standard_normal(len(e)) if couplings == "gauss" else rng.choice([-1.0, 1.0], len(e))
    return IsingInstance(L * L, e, Je, beta_J, f"planarEA_{couplings}_L{L}_b{beta_J}_s{seed}")


# =============================================================================== solvers of the moving-field Ising model
class ExactSolver:
    """Exhaustive enumeration (N <= ~20): exact posterior P_t(y|x) for fields H (B,N)."""

    def __init__(self, inst):
        self.inst, n = inst, inst.N; self.N = n
        idx = np.arange(2 ** n, dtype=np.int64)
        self.X = (1 - 2 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)).astype(np.float32)
        lw = inst.beta_J * inst.energy(self.X); self.logZ0 = logsumexp(lw); self.logmu = lw - self.logZ0

    def logpost(self, H):
        return self.logmu[None] + H.astype(np.float32) @ self.X.T

    def weights(self, H):
        lp = self.logpost(H); return np.exp(lp - logsumexp(lp, 1, keepdims=True))

    def logZ(self, H):
        """log Z_J(h)/Z_J(0) = log sum_y mu(y) exp(h.y)."""
        return logsumexp(self.logpost(H), 1)

    def yhat(self, H, chunk=64):
        return np.concatenate([self.weights(H[i:i + chunk]) @ self.X for i in range(0, len(H), chunk)])

    def sample_prior(self, n, rng):
        p = np.exp(self.logmu); p /= p.sum(); return self.X[rng.choice(len(p), n, p=p)].astype(float)

    def sample(self, H, K, rng, chunk=64):
        out = np.empty((len(H), K, self.N))
        for i in range(0, len(H), chunk):
            w = self.weights(H[i:i + chunk]); cu = np.cumsum(w, 1)
            s = np.array([np.searchsorted(c, rng.random(K)) for c in cu])
            out[i:i + chunk] = self.X[np.minimum(s, len(self.X) - 1)]
        return out

    def overlap_pmf(self, H, Y0, chunk=32):
        n = self.N; out = np.empty((len(H), n + 1))
        for i in range(0, len(H), chunk):
            w = self.weights(H[i:i + chunk]); k = ((Y0[i:i + chunk].astype(np.float32) @ self.X.T + n) / 2).round().astype(np.int64)
            for r in range(len(w)): out[i + r] = np.bincount(k[r], weights=w[r], minlength=n + 1)
        return out

    def chi_sg(self, H, chunk=16):
        res = []
        for i in range(0, len(H), chunk):
            w = self.weights(H[i:i + chunk]).astype(np.float32); m = w @ self.X
            C = np.stack([self.X.T @ (ww[:, None] * self.X) for ww in w]) - m[:, :, None] * m[:, None, :]
            res.append((C ** 2).sum((1, 2)) / self.N)
        return np.concatenate(res)


class ProductMixtureSolver:
    """mu(y) = sum_a W_a prod_i (1 + m_ai y_i)/2 : an exactly solvable ultrametric caricature."""

    def __init__(self, m, W, labels=None):
        self.m = np.asarray(m, float); self.A, self.N = self.m.shape; self.W = np.asarray(W, float) / np.sum(W); self.labels = labels
        self.th = np.arctanh(np.clip(self.m, -0.999999, 0.999999))

    def _tilt(self, H):
        u = self.th[None] + H[:, None, :]
        logz = (np.logaddexp(u, -u) - np.logaddexp(self.th, -self.th)[None]).sum(-1) + np.log(self.W)[None]
        return logz, np.tanh(u)

    def logZ(self, H, chunk=200):
        return np.concatenate([logsumexp(self._tilt(H[i:i + chunk])[0], 1) for i in range(0, len(H), chunk)])

    def post_states(self, H):
        logz, _ = self._tilt(H); return np.exp(logz - logsumexp(logz, 1, keepdims=True))

    def yhat(self, H, chunk=200):
        out = []
        for i in range(0, len(H), chunk):
            logz, mh = self._tilt(H[i:i + chunk]); p = np.exp(logz - logsumexp(logz, 1, keepdims=True))
            out.append(np.einsum("ba,ban->bn", p, mh))
        return np.concatenate(out)

    def chi_sg(self, H, chunk=200):
        """Exact (1/N) Tr C^2 of the moving-field posterior (law of total covariance, low-rank form)."""
        out = []
        for i in range(0, len(H), chunk):
            logz, mh = self._tilt(H[i:i + chunk]); p = np.exp(logz - logsumexp(logz, 1, keepdims=True))
            m = np.einsum("ba,ban->bn", p, mh); d = np.einsum("ba,ban->bn", p, 1 - mh ** 2)
            Bv = np.sqrt(p)[..., None] * (mh - m[:, None, :]); G = np.einsum("ban,bcn->bac", Bv, Bv)
            out.append(((d ** 2).sum(-1) + 2 * (d * (Bv ** 2).sum(1)).sum(-1) + (G ** 2).sum((1, 2))) / self.N)
        return np.concatenate(out)

    def sample_prior(self, n, rng):
        a = rng.choice(self.A, n, p=self.W); return np.where(rng.random((n, self.N)) < (1 + self.m[a]) / 2, 1.0, -1.0)

    def sample(self, H, K, rng, chunk=200):
        out = []
        for i in range(0, len(H), chunk):
            logz, mh = self._tilt(H[i:i + chunk]); p = np.exp(logz - logsumexp(logz, 1, keepdims=True)); cu = p.cumsum(1)
            a = np.minimum((rng.random((len(p), K, 1)) > cu[:, None, :]).sum(-1), self.A - 1)
            pr = (1 + mh[np.arange(len(p))[:, None], a]) / 2; out.append(np.where(rng.random(pr.shape) < pr, 1.0, -1.0))
        return np.concatenate(out)


def ultrametric_caricature(N=500, branching=(2, 3, 3), m0=0.9, flips=(0.5, 0.15, 0.05), mirror=True, seed=7):
    """Tree of state magnetisations (independent sign patterns at level 1, then flips of a fraction of the
    parent's signs), magnetisation m0, Dirichlet(1) weights at every level, optional mirror states."""
    rng = np.random.default_rng(seed); root = np.where(rng.random(N) < 0.5, 1.0, -1.0)
    vecs, labs, ws = [root], [()], [1.0]
    for l, b in enumerate(branching):
        nv, nl, nw = [], [], []
        for v, lab, w in zip(vecs, labs, ws):
            d = rng.dirichlet(np.ones(b))
            for c in range(b):
                nv.append(np.where(rng.random(N) < flips[l], -v, v)); nl.append(lab + (c,)); nw.append(w * d[c])
        vecs, labs, ws = nv, nl, nw
    m, W, labs = m0 * np.array(vecs), np.array(ws), np.array(labs)
    if mirror:
        m = np.concatenate([m, -m]); W = np.concatenate([W, W]) / 2
        labs = np.concatenate([np.column_stack([np.zeros(len(labs), int), labs]), np.column_stack([np.ones(len(labs), int), labs])])
    return ProductMixtureSolver(m, W, labs)


class PoolSolver:
    """Moving-field posterior approximated by reweighting an independent equilibrium pool by exp(h.y).
    Exact for the empirical measure; reliable while the effective sample size is large."""

    def __init__(self, pool):
        self.X = np.asarray(pool, np.float32); self.M, self.N = self.X.shape

    def weights(self, H):
        lw = H.astype(np.float32) @ self.X.T; lw -= lw.max(1, keepdims=True); w = np.exp(lw.astype(np.float64))
        return w / w.sum(1, keepdims=True)

    def yhat(self, H, chunk=128):
        return np.concatenate([self.weights(H[i:i + chunk]) @ self.X for i in range(0, len(H), chunk)])

    def ess(self, H):
        w = self.weights(H); return 1.0 / (w ** 2).sum(1)

    def sample(self, H, K, rng):
        w = self.weights(H); cu = np.cumsum(w, 1)
        s = np.array([np.searchsorted(c, rng.random(K)) for c in cu]); return self.X[np.minimum(s, self.M - 1)].astype(float)


class BPSolver:
    """Belief propagation for the moving-field Ising model on the instance graph (exact on trees,
    accurate on locally tree-like graphs in single-state regimes)."""

    def __init__(self, inst, damping=0.5, n_iter=300, tol=1e-7):
        self.inst, self.N = inst, inst.N; self.damping, self.n_iter, self.tol = damping, n_iter, tol
        e = inst.edges; self.src = np.r_[e[:, 0], e[:, 1]]; self.dst = np.r_[e[:, 1], e[:, 0]]
        self.tJ = np.tanh(inst.beta_J * np.r_[inst.Je, inst.Je]); E = len(e)
        self.rev = np.r_[np.arange(E, 2 * E), np.arange(E)]

    def yhat(self, H):
        B = len(H); u = np.zeros((B, len(self.src)))
        for it in range(self.n_iter):
            tot = H.copy(); np.add.at(tot.T, self.dst, u.T)               # total field at each site
            cav = tot[:, self.src] - u[:, self.rev]                       # cavity field of src excluding dst
            new = np.arctanh(np.clip(self.tJ * np.tanh(cav), -1 + 1e-12, 1 - 1e-12))
            d = np.abs(new - u).max(); u = self.damping * u + (1 - self.damping) * new
            if d < self.tol: break
        tot = H.copy(); np.add.at(tot.T, self.dst, u.T); self.iters = it + 1
        return np.tanh(tot)


def pt_sample(inst, betas, n_chains, n_sweeps, burn, every, rng, field=None):
    """Parallel tempering of mu (or of the moving-field model if field (C,N) is given, identical across
    replicas) with graph-colour parallel Metropolis updates.  Returns samples at the largest beta
    (n_kept, C, N) as int8 and swap acceptance."""
    betas = np.asarray(betas, float); R, N, C = len(betas), inst.N, n_chains
    nbr, Jn = inst.nbr, inst.Jn
    x = np.where(rng.random((R, C, N + 1)) < 0.5, 1.0, -1.0); x[..., N] = 0.0          # dummy site N
    hf = np.zeros((C, N)) if field is None else field
    E = 0.5 * np.einsum("rcn,nd,rcnd->rc", x[..., :N], Jn, x[..., nbr]); out = []; acc = np.zeros(R - 1)
    for sw in range(n_sweeps):
        for cls in inst.colors:
            f = np.einsum("rcnd,nd->rcn", x[..., nbr[cls]], Jn[cls]); xi = x[..., cls]
            dE = -2 * xi * f; dl = betas[:, None, None] * dE - 2 * xi * hf[None, :, cls]
            a = np.log(rng.random(dl.shape) + 1e-300) < dl
            x[..., cls] = np.where(a, -xi, xi); E += np.where(a, dE, 0).sum(-1)
        for r in range(sw % 2, R - 1, 2):
            sw_ = np.log(rng.random(C) + 1e-300) < (betas[r] - betas[r + 1]) * (E[r + 1] - E[r])
            if sw_.any():
                x[r, sw_], x[r + 1, sw_] = x[r + 1, sw_].copy(), x[r, sw_].copy(); E[r, sw_], E[r + 1, sw_] = E[r + 1, sw_].copy(), E[r, sw_].copy()
            acc[r] += sw_.mean()
        if sw >= burn and (sw - burn) % every == 0: out.append(x[-1, :, :N].astype(np.int8).copy())
    return np.array(out), acc / max(n_sweeps // 2, 1)


# =============================================================================== observables
def order_parameter_paths(solver, bridge, Y, tgrid, rng, n_sub=40):
    """Planted H-PID trajectories ending at the configurations Y (B,N): x(t) on tgrid, order parameter
    yhat(t) = E[y|x(t)], overlap clock q(t) = |yhat|^2/N and overlap with the truth yhat.y/N."""
    X = bridge.bridge_paths(Y, tgrid, rng, n_sub=n_sub); B, T, N = X.shape
    Yh = np.stack([solver.yhat(bridge.kappa(t) * X[:, k]) for k, t in enumerate(tgrid)], 1)
    return dict(X=X, Yh=Yh, q=(Yh ** 2).sum(-1) / N, q_truth=np.einsum("btn,bn->bt", Yh, Y) / N)


def branching_overlaps(solver, bridge, X_at, t_s, Y, K, rng):
    """Endpoints of K trajectories branching at t_s from states X_at (B,N): y' ~ P_{t_s}(.|x).
    Returns overlaps with the original endpoints (B,K) and among branches (B, K(K-1)/2)."""
    Yb = solver.sample(bridge.kappa(t_s) * X_at, K, rng); N = Y.shape[1]
    Qo = np.einsum("bkn,bn->bk", Yb, Y) / N
    iu = np.triu_indices(K, 1); G = np.einsum("bkn,bln->bkl", Yb, Yb) / N
    return Qo, G[:, iu[0], iu[1]], Yb


def ball_tails(pmf, N):
    """signed P(Q>=q) on the lattice and |Q| tails P(|Q|>=q) for q>=0."""
    qv = np.linspace(-1, 1, N + 1); sgn = np.cumsum(pmf[:, ::-1], 1)[:, ::-1]
    qa = qv[qv >= -1e-9]; ab = np.stack([pmf[:, np.abs(qv) >= q - 1e-9].sum(1) for q in qa], 1)
    return qv, sgn, qa, ab


def crossing(t, rho, lam):
    for k in range(1, len(t)):
        if rho[k - 1] < lam <= rho[k]:
            f = (lam - rho[k - 1]) / (rho[k] - rho[k - 1]); return float(t[k - 1] + f * (t[k] - t[k - 1]))
    return float("nan")


def energy_information_check(solver, bridge, Y, tgrid, rng):
    """beta = 0: (1/2) int_0^t E|u|^2 ds  versus  -(N/2) log(1-t) - I(y; x_t), with exact log p_t from logZ."""
    assert bridge.beta == 0.0
    N = Y.shape[1]; X = bridge.bridge_paths(Y, tgrid, rng)
    E_u2, I = [], []
    for k, t in enumerate(tgrid):
        x = X[:, k]; kap = bridge.kappa(t); yh = solver.yhat(kap * x)
        E_u2.append(np.mean(((kap * yh - bridge.c(t) * x) ** 2).sum(1)))
        s2, a = bridge.sigma2(t), bridge.a(t)
        logp_xy = -((x - a * Y) ** 2).sum(1) / (2 * s2) - N / 2 * np.log(2 * np.pi * s2)
        logp_x = -(x ** 2).sum(1) / (2 * s2) - N * a ** 2 / (2 * s2) - N / 2 * np.log(2 * np.pi * s2) + solver.logZ(kap * x)
        I.append(np.mean(logp_xy - logp_x))
    E_u2 = np.array(E_u2); tt = np.r_[0.0, tgrid]
    u0 = float((solver.yhat(np.zeros((1, N)))[0] ** 2).sum())          # u(0,0) = E[y]
    e2 = np.r_[u0, E_u2]; cum = np.cumsum(0.5 * 0.5 * (e2[1:] + e2[:-1]) * np.diff(tt))
    return dict(t=list(map(float, tgrid)), energy=cum.tolist(), I=list(map(float, I)),
                prediction=(-N / 2 * np.log(1 - np.asarray(tgrid)) - np.array(I)).tolist(), E_u2=E_u2.tolist())
