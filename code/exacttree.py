"""Exact enumeration of moving-field Ising posteriors up to N~24, with
(i) explicit Z2 gauge (sector = sign of overlap with the ground state),
(ii) a NON-spectral hierarchy: zero-temperature valleys grouped by overlap (average linkage),
    split top-down with a mass and overlap-gap acceptance rule; depth chosen by the data,
(iii) level-resolved class information along planted observation paths H_g = g Y + B_g.

The configuration space is enumerated as an outer product of two halves, y=(ya,yb), so that
log-weights and field terms are 2^(N/2) x 2^(N/2) arrays; this keeps N=24 cheap.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.cluster.hierarchy import linkage

import numba as nb


@nb.njit(fastmath=True, cache=True)
def _rowmax(logp, A, B, i):
    m = -1e300
    for j in range(logp.shape[1]):
        v = logp[i, j] + A[i] + B[j]
        m = max(m, v)
    return m


@nb.njit(fastmath=True, cache=True)
def _chunk(logp, A, B, lab, Mx, i0, i1, rows, cols, cls):
    for i in range(i0, i1):
        acc = 0.0
        for j in range(logp.shape[1]):
            w = np.exp(logp[i, j] + A[i] + B[j] - Mx)
            acc += w
            cols[j] += w
            cls[lab[i, j]] += w
        rows[i] = acc


@nb.njit(parallel=True, cache=True)
def _field_pass(logp, A, B, lab, K, nchunk):
    na, nbb = logp.shape
    rm = np.empty(na)
    for i in nb.prange(na):
        rm[i] = _rowmax(logp, A, B, i)
    Mx = rm.max()
    rows = np.zeros(na); cols = np.zeros((nchunk, nbb)); cls = np.zeros((nchunk, K))
    step = (na + nchunk - 1) // nchunk
    for c in nb.prange(nchunk):
        _chunk(logp, A, B, lab, Mx, c * step, min(na, (c + 1) * step), rows, cols[c], cls[c])
    return rows, cols.sum(0), cls.sum(0)


S_STAR, S10, S90 = 1.044, 0.1488, 3.759     # binary-symmetric Gaussian channel: 50/10/90 % of log 2


def half_configs(n):
    idx = np.arange(2 ** n)[:, None]
    return (1 - 2 * ((idx >> np.arange(n)) & 1)).astype(np.float32)   # (2^n, n), +-1


class ExactModel:
    """Gibbs measure mu(y) ~ exp(1/2 y^T J y) on {-1,1}^N, enumerated exactly."""

    def __init__(self, J, name=""):
        J = np.asarray(J, float); J = 0.5 * (J + J.T); np.fill_diagonal(J, 0.0)
        self.J, self.N, self.name = J, J.shape[0], name
        N = self.N; self.na = N // 2; self.nb = N - self.na
        self.Ya, self.Yb = half_configs(self.na), half_configs(self.nb)
        Ja, Jb, Jab = J[:self.na, :self.na], J[self.na:, self.na:], J[:self.na, self.na:]
        Ea = 0.5 * np.einsum('ki,ij,kj->k', self.Ya, Ja, self.Ya)
        Eb = 0.5 * np.einsum('ki,ij,kj->k', self.Yb, Jb, self.Yb)
        L = Ea[:, None] + Eb[None, :] + (self.Ya @ Jab.astype(np.float32)) @ self.Yb.T
        L = L.astype(np.float64); L -= logsumexp(L)
        self.logp = L                                  # (2^na, 2^nb)
        self.p = np.exp(L)
        self.C0 = self.second_moment(self.p)           # zero-field: <y>=0, so C0 = <y y^T>
        self.lam = np.sort(np.linalg.eigvalsh(self.C0))[::-1]

    # ---------------------------------------------------------------- helpers
    def config(self, flat):
        a, b = np.divmod(np.asarray(flat), 2 ** self.nb)
        return np.concatenate([self.Ya[a], self.Yb[b]], axis=-1)

    def dot_grid(self, v):
        v = np.asarray(v, np.float32)
        return (self.Ya @ v[:self.na])[:, None] + (self.Yb @ v[self.na:])[None, :]

    def first_moment(self, W):
        return np.concatenate([self.Ya.T @ W.sum(1), self.Yb.T @ W.sum(0)])

    def second_moment(self, W):
        Wa, Wb = W.sum(1), W.sum(0)
        Saa = (self.Ya * Wa[:, None]).T @ self.Ya
        Sbb = (self.Yb * Wb[:, None]).T @ self.Yb
        Sab = self.Ya.T @ (W.astype(np.float32) @ self.Yb)
        return np.block([[Saa, Sab], [Sab.T, Sbb]]).astype(np.float64)

    def sample(self, n, rng):
        c = np.cumsum(self.p.ravel()); c /= c[-1]
        return self.config(np.searchsorted(c, rng.random(n)))

    def fast(self, h, lab, K):
        """Posterior row/col marginals and class masses (normalized) without forming the grid."""
        h = np.asarray(h, np.float64)
        A = self.Ya.astype(np.float64) @ h[:self.na]; B = self.Yb.astype(np.float64) @ h[self.na:]
        if not hasattr(self, 'logp32'): self.logp32 = self.logp.astype(np.float32)
        r, c, k = _field_pass(self.logp32, A, B, lab, K, 8)
        Z = r.sum(); r /= Z; c /= Z; k /= Z
        m = np.concatenate([self.Ya.T.astype(np.float64) @ r, self.Yb.T.astype(np.float64) @ c])
        return m, k

    def posterior(self, h):
        s = self.logp + self.dot_grid(h)
        s -= s.max(); W = np.exp(s); W /= W.sum(); return W

    # ---------------------------------------------------------------- Z2 gauge and valleys
    def gauge(self):
        """sector(y) = sign(y.g), g = ground state; ties broken by y_1. Returns +-1 grid."""
        flat = int(np.argmax(self.logp)); self.g = self.config(flat)
        ov = self.dot_grid(self.g)
        tie = np.where(self.Ya[:, :1] > 0, 1.0, -1.0) * np.ones((1, len(self.Yb)), np.float32)
        self.sector = np.where(ov > 0, 1, np.where(ov < 0, -1, tie)).astype(np.int8)
        return self.sector

    def descend(self, Y):
        """Greedy zero-temperature single-spin-flip descent (maximizes y^T J y)."""
        Y = Y.copy().astype(float)
        for _ in range(10 * self.N):
            loc = Y * (Y @ self.J)
            i = np.argmin(loc, 1); worst = loc[np.arange(len(Y)), i]
            move = worst < -1e-12
            if not move.any(): break
            Y[np.where(move)[0], i[move]] *= -1
        return Y

    def valleys(self, cover=0.97, kmax=6000, mmin_valley=0.002):
        """Quotient (gauge-fixed) configurations covering `cover` of the mass are descended to their
        zero-T minima; minima are identified up to global flip. Returns minima (M,N) and masses."""
        if not hasattr(self, 'sector'): self.gauge()
        pq = np.where(self.sector > 0, 2 * self.p, 0.0).ravel()        # quotient mass on representatives
        order = np.argsort(-pq); cum = np.cumsum(pq[order])
        k = int(min(kmax, np.searchsorted(cum, cover) + 1)); top = order[:k]
        self.valley_cover = float(cum[k - 1])
        Ymin = self.descend(self.config(top))
        Ymin *= np.where(Ymin @ self.g >= 0, 1, -1)[:, None]            # gauge-fix minima
        key = (Ymin > 0) @ (1 << np.arange(self.N, dtype=np.int64))
        uk, inv = np.unique(key, return_inverse=True)
        mass = np.bincount(inv, weights=pq[top]); mins = np.array([Ymin[np.argmax(inv == j)] for j in range(len(uk))])
        keep = mass >= mmin_valley * mass.sum()
        if keep.sum() < 2: keep = np.argsort(-mass)[:2]
        return mins[keep], mass[keep]


def _kmeans2(M, wq, mask, inits, iters=40):
    """Exact mass-weighted 2-means (Lloyd) on the gauge representatives inside `mask`.
    Returns the best split (max between-class variance) as a boolean grid and its stats."""
    best = None; N = M.N
    for m1, m2 in inits:
        prev = None
        for _ in range(iters):
            d = m1 - m2; b = 0.5 * (m1 @ m1 - m2 @ m2)
            side = (M.dot_grid(d) > b) & mask
            if prev is not None and np.array_equal(side, prev): break
            prev = side
            w1 = (wq * side); w2 = (wq * (mask & ~side)); s1, s2 = w1.sum(), w2.sum()
            if s1 <= 0 or s2 <= 0: break
            m1 = M.first_moment(w1) / s1; m2 = M.first_moment(w2) / s2
        w1 = (wq * side); w2 = (wq * (mask & ~side)); s1, s2 = w1.sum(), w2.sum()
        if s1 <= 0 or s2 <= 0: continue
        m1 = M.first_moment(w1) / s1; m2 = M.first_moment(w2) / s2
        mp = (s1 * m1 + s2 * m2) / (s1 + s2)
        D2 = (s1 * ((m1 - mp) @ (m1 - mp)) + s2 * ((m2 - mp) @ (m2 - mp))) / N
        if best is None or D2 > best[0]: best = (D2, side.copy(), s1, s2, m1, m2)
    return best


def build_tree(model, mmin=0.03, gap=0.10, kappa_min=2.0, max_levels=3, restarts=6, seed=0):
    """Nested partition of the exact Gibbs measure.
    Level 0: the Z2 sector, sign of the overlap with the ground state.
    Glassy levels: inside a sector (gauge representatives), each class is split by exact
    mass-weighted 2-means (several initializations, best between-class variance); the split is
    accepted if both children carry >= mmin of the total mass, the mean within-child overlap
    exceeds the between-child overlap by >= gap, and the split is collective: its between-class
    variance kappa = p1 p2 |m1-m2|^2 (parent-normalized) is >= kappa_min single-spin units, so that by
    the binary-channel law it is decided before local spins (gamma ~ 1). Depth is decided by the data (<= max_levels).
    Each accepted split is annotated by the zero-temperature valleys found on each side."""
    M = model; N = M.N; rng = np.random.default_rng(seed)
    sec = M.gauge(); rep = sec > 0; wq = np.where(rep, 2 * M.p, 0.0)
    mins, vmass = M.valleys()
    cls = np.where(rep, 0, -1).astype(np.int16)
    levels = [cls.copy()]; info = []; cand = []
    for lev in range(max_levels):
        K = int(cls.max()) + 1; new = cls.copy(); nxt = 0; splits = []
        for c in range(K):
            mask = cls == c; wc = wq * mask; mc = wc.sum()
            inits = []
            m = M.first_moment(wc) / mc; C = M.second_moment(wc) / mc - np.outer(m, m)
            u = np.linalg.eigh(C)[1][:, -1]
            sd = np.sqrt(max(u @ C @ u, 1e-12)); inits.append((m + sd * u, m - sd * u))
            vin = [v for v in range(len(mins)) if mask[tuple(_idx(M, mins[v]))]]
            if len(vin) >= 2:
                o = np.argsort(-vmass[vin]); inits.append((mins[vin[o[0]]], mins[vin[o[1]]]))
            fl = np.flatnonzero(mask.ravel()); pr = wc.ravel()[fl]; pr = pr / pr.sum()
            for _ in range(restarts):
                a, b = rng.choice(fl, 2, p=pr); inits.append((M.config(a), M.config(b)))
            res = _kmeans2(M, wq, mask, inits) if mc >= 2 * mmin else None
            ok = False
            if res is not None:
                D2, side, s1, s2, m1, m2 = res
                qin = (s1 * (m1 @ m1) + s2 * (m2 @ m2)) / (s1 + s2) / N; qbt = (m1 @ m2) / N
                kap = s1 * s2 / (s1 + s2) ** 2 * ((m1 - m2) @ (m1 - m2))
                ok = s1 >= mmin and s2 >= mmin and qin - qbt >= gap and kap >= kappa_min
                cand.append(dict(level=lev + 1, parent=c, mass=float(mc), masses=[float(s1), float(s2)],
                                 gap=float(qin - qbt), kappa=float(kap), accepted=bool(ok)))
            if ok:
                new[mask & side] = nxt; new[mask & ~side] = nxt + 1; nxt += 2
                vl = [v for v in vin if side[tuple(_idx(M, mins[v]))]]; vr = [v for v in vin if v not in vl]
                splits.append(dict(parent=c, mass=float(mc), masses=[float(s1), float(s2)], gap=float(qin - qbt),
                                   qin=float(qin), qbt=float(qbt), D2=float(D2), kappa=float(kap),
                                   valleys=[float(vmass[vl].sum() / vmass.sum()), float(vmass[vr].sum() / vmass.sum())]))
            else:
                new[mask] = nxt; nxt += 1
        if not splits: break
        cls = new; levels.append(cls.copy()); info.append(splits)
    flipped = lambda g: g[::-1, ::-1]
    labels, ncls = [], []
    for g in levels:
        full = np.where(rep, g, flipped(g)); K = int(full.max()) + 1
        labels.append((np.where(rep, 1, 0) * K + full).astype(np.int16)); ncls.append(2 * K)
    return dict(labels=labels, n_classes=ncls, level_info=info, candidates=cand, valleys=mins, valley_mass=vmass / vmass.sum(),
                cover=M.valley_cover)


def _idx(M, y):
    a = int(((y[:M.na] < 0) * (1 << np.arange(M.na))).sum()); b = int(((y[M.na:] < 0) * (1 << np.arange(M.nb))).sum())
    return a, b


def class_masses(W, lab, K):
    return np.bincount(lab.ravel(), weights=W.ravel(), minlength=K)


def H(p):
    p = np.asarray(p); return float(-(p[p > 0] * np.log(p[p > 0])).sum())


def stratified_endpoints(M, lab, K, npaths, rng):
    """Endpoints Y stratified over the classes of `lab` (finest level): n_k ~ max(2, npaths*pi_k) draws
    from mu(.|k); returns Y and weights w_i = pi_k/n_k (sum to one)."""
    flat = lab.ravel(); P = M.p.ravel(); pis = np.bincount(flat, weights=P, minlength=K)
    n = np.maximum(2, np.round(npaths * pis).astype(int)); n[pis <= 0] = 0
    Ys, ws = [], []
    for k in range(K):
        if n[k] == 0: continue
        idx = np.flatnonzero(flat == k); c = np.cumsum(P[idx]); c /= c[-1]
        Ys.append(M.config(idx[np.searchsorted(c, rng.random(n[k]))])); ws.append(np.full(n[k], pis[k] / n[k]))
    Y = np.concatenate(Ys); w = np.concatenate(ws); perm = rng.permutation(len(Y))
    return Y[perm], w[perm] / w.sum()


def run_paths(M, tree, gammas, npaths, rng, cov_paths=0):
    """Planted paths H_g = g Y + B_g with endpoints stratified over the finest classes (weights w);
    exact posterior class masses at each level; optionally the posterior covariance spectrum."""
    labs, Ks = tree['labels'], tree['n_classes']; L = len(labs)
    Y, w = stratified_endpoints(M, labs[-1], Ks[-1], npaths, rng); npaths = len(Y)
    dg = np.diff(np.concatenate([[0.0], gammas]))
    B = np.cumsum(rng.standard_normal((npaths, len(gammas), M.N)) * np.sqrt(dg)[None, :, None], 1)
    Hs = gammas[None, :, None] * Y[:, None, :] + B
    a = ((Y[:, :M.na] < 0) @ (1 << np.arange(M.na))).astype(int); b = ((Y[:, M.na:] < 0) @ (1 << np.arange(M.nb))).astype(int)
    plant = [lab[a, b] for lab in labs]
    post = [np.zeros((npaths, len(gammas), K)) for K in Ks]
    fin = labs[-1]; Kf = Ks[-1]; fl = fin.ravel()
    rep = np.array([np.argmax(fl == c) for c in range(Kf)])
    maps = [lab.ravel()[rep] for lab in labs]
    qself = np.zeros((npaths, len(gammas)))
    topvec = np.zeros((cov_paths, len(gammas), M.N)); toplam = np.zeros((cov_paths, len(gammas), 3))
    for i in range(npaths):
        for k, g in enumerate(gammas):
            m, kf = M.fast(Hs[i, k], fin, Kf)
            for l in range(L): post[l][i, k] = np.bincount(maps[l], weights=kf, minlength=Ks[l])
            qself[i, k] = m @ m / M.N
            if i < cov_paths:
                W = M.posterior(Hs[i, k])
                C = M.second_moment(W) - np.outer(m, m); ev, U = np.linalg.eigh(C)
                topvec[i, k] = U[:, -1]; toplam[i, k] = ev[::-1][:3]
    return dict(Y=Y, w=w, plant=plant, post=post, qself=qself, topvec=topvec, toplam=toplam, H=Hs)


def level_information(M, tree, res):
    """Class information I(C_l;H) and conditional increments, normalized by the exact terminal entropy."""
    labs, Ks = tree['labels'], tree['n_classes']
    prior = [class_masses(M.p, lab, K) for lab, K in zip(labs, Ks)]
    I = []
    for l, K in enumerate(Ks):
        P = np.clip(res['post'][l], 1e-300, 1)
        w = res.get('w', np.full(P.shape[0], 1.0 / P.shape[0]))
        Hpost = (w[:, None] * -(P * np.log(P)).sum(-1)).sum(0)
        I.append(H(prior[l]) - Hpost)
    I = np.array(I); Hl = np.array([H(p) for p in prior])
    dI = np.diff(np.concatenate([[np.zeros_like(I[0])], I]), axis=0)     # conditional increments
    dH = np.diff(np.concatenate([[0.0], Hl]))
    return I, dI, Hl, dH


def half_time(g, frac):
    g = np.asarray(g); f = np.asarray(frac)
    for k in range(1, len(g)):
        if f[k] >= 0.5 and g[k - 1] > 0:
            x0, x1 = np.log(g[k - 1]), np.log(g[k]); y0, y1 = f[k - 1], f[k]
            return float(np.exp(x0 + (0.5 - y0) * (x1 - x0) / max(y1 - y0, 1e-12)))
        if f[k] >= 0.5: return float(g[k])
    return np.nan


def conditional_lambdas(M, tree):
    """For each level l>=1: top eigenvalue of the parent-conditional covariance, averaged over the
    parents that split (mass-weighted in log). Level 0 (Z2): top eigenvalue of C0."""
    labs, Ks = tree['labels'], tree['n_classes']; out = [M.lam[0]]
    for l in range(1, len(labs)):
        par, ch = labs[l - 1], labs[l]
        logs, ws = [], []
        for c in range(Ks[l - 1]):
            Wp = M.p * (par == c); mp = Wp.sum()
            if mp <= 0: continue
            kids = np.unique(ch[par == c])
            if len(kids) < 2: continue
            Wn = Wp / mp; m = M.first_moment(Wn); C = M.second_moment(Wn) - np.outer(m, m)
            logs.append(np.log(np.linalg.eigvalsh(C)[-1])); ws.append(mp)
        out.append(float(np.exp(np.average(logs, weights=ws))) if logs else np.nan)
    return np.array(out)


def parent_channels(M, tree):
    """Per level l>=0 and per splitting parent P: (pi_P, H(children|P), kappa_P = top eigenvalue of Cov(Y|P))."""
    labs, Ks = tree['labels'], tree['n_classes']
    out = [[dict(pi=1.0, H=float(np.log(2)), kappa=float(M.lam[0]))]]
    for l in range(1, len(labs)):
        par, ch = labs[l - 1], labs[l]; rows = []
        for c in range(Ks[l - 1]):
            Wp = M.p * (par == c); mp = Wp.sum()
            if mp <= 0: continue
            km = np.bincount(ch[par == c].ravel(), weights=Wp[par == c].ravel(), minlength=Ks[l]); km = km[km > 0] / mp
            if len(km) < 2: continue
            Wn = Wp / mp; m = M.first_moment(Wn); C = M.second_moment(Wn) - np.outer(m, m)
            rows.append(dict(pi=float(mp), H=H(km), kappa=float(np.linalg.eigvalsh(C)[-1])))
        out.append(rows)
    return out


def contrasts(M, tree, level, parent_class):
    """Gauge-free contrast vector between the two children of `parent_class` at `level`-1."""
    par, ch = tree['labels'][level - 1], tree['labels'][level]
    kids = np.unique(ch[par == parent_class]); ms = []
    for c in kids[:2]:
        Wc = M.p * (ch == c); ms.append(M.first_moment(Wc) / Wc.sum())
    return ms[0] - ms[1]


# ---------------------------------------------------------------- instance families
def bethe_J(N, beta, seed, d=3):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        stubs = np.repeat(np.arange(N), d); rng.shuffle(stubs); e = stubs.reshape(-1, 2)
        if np.any(e[:, 0] == e[:, 1]): continue
        k = np.sort(e, 1)
        if len(np.unique(k[:, 0] * N + k[:, 1])) == len(e): break
    Je = rng.standard_normal(len(k)); J = np.zeros((N, N)); J[k[:, 0], k[:, 1]] = Je; J += J.T
    return beta * J


def sk_J(N, beta, seed):
    rng = np.random.default_rng(10_000 + seed)
    A = rng.standard_normal((N, N)); J = np.triu(A, 1); J = J + J.T
    return beta * J / np.sqrt(N)


def hopfield_tree_J(N, beta, seed, branching=(2, 2, 2), flips=(None, 4, 2), weights=None):
    """Pairwise Hopfield model with tree-correlated patterns (Parga-Virasoro type).
    Root children: independent random patterns; each deeper child copies its parent and flips
    `flips[l]` randomly chosen spins. J = (beta/N) sum_mu w_mu xi^mu xi^mu^T."""
    rng = np.random.default_rng(seed)
    level = [rng.choice([-1.0, 1.0], N) for _ in range(branching[0])]
    tree_levels = [level]
    for l in range(1, len(branching)):
        nxt = []
        for p in tree_levels[-1]:
            for _ in range(branching[l]):
                c = p.copy(); idx = rng.choice(N, flips[l], replace=False); c[idx] *= -1; nxt.append(c)
        tree_levels.append(nxt)
    xi = np.array(tree_levels[-1])
    w = np.ones(len(xi)) if weights is None else np.asarray(weights)
    J = beta / N * (xi.T * w) @ xi
    return J, tree_levels
