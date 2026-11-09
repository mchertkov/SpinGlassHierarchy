"""Stage 6B analysis: do the hidden units of a learned PID denoiser represent the frozen hierarchy?

For every trained network we measure, using ONLY quantities defined independently of training:
 (a) unit geometry   - cosine of each unit's input weight w_k with the coarse contrast
                       d_c = m_A - m_B, the within-parent fine contrasts d_f[p] = m_{p1} - m_{p2},
                       and the leaf means; fraction of ||w_k||^2 in the hierarchy plane;
 (b) unit decisions  - plug-in (Miller-Madow corrected) information I(C0; a_k) and
                       I(C1; a_k | C0) as functions of gamma; unit half-times;
                       a linear probe on the whole hidden layer (lower bound on I(C; a));
 (c) linear response - the top eigenvector of the symmetrised denoiser Jacobian (the learned
                       posterior covariance C_t = d m / d h) along planted paths, and its
                       alignment with d_c and with the planted parent's d_f; exact at N=16;
 (d) importance      - mean-ablation increase of test MSE, averaged over log-gamma.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from e6_common import (OUT, ExactHierarchy, load_library, null_sample, frozen_large_tree, frozen_null_tree,
                       hierarchy_geometry, predict, entropy, json_dump, brownian_fields)
from e6b_train import load_net, Source

GU = np.array([.005, .01, .02, .04, .06, .085, .12, .16, .21, .28, .36, .46, .58, .72, .9, 1.15, 1.45, 1.85, 2.4, 3.2])
NTEST = 4000


# ----------------------------------------------------------------------------- targets
_GEOM = {}


def target(kind, N, ident):
    key = (kind, N, ident)
    if key in _GEOM: return _GEOM[key]
    rng = np.random.default_rng(31337 + 11 * ident + N)
    if kind == 'bethe16':
        E = ExactHierarchy('bethe', ident); tree = E.tree
        geo = dict(pi=E.pi, means=E.leaf_means)
        par_pi = E.pi.reshape(2, 2).sum(1)
        pm = np.stack([(E.pi[2 * p] * E.leaf_means[2 * p] + E.pi[2 * p + 1] * E.leaf_means[2 * p + 1]) / par_pi[p] for p in range(2)])
        geo.update(par_pi=par_pi, par_means=pm, d_coarse=pm[0] - pm[1],
                   d_fine=np.stack([E.leaf_means[2 * p] - E.leaf_means[2 * p + 1] for p in range(2)]))
        draw = lambda n, r: E.sample_prior(n, r)
        extra = dict(E=E)
    elif kind == 'bethe':
        Y, z = load_library(N, ident); tree = frozen_large_tree(N, ident)
        geo = hierarchy_geometry(Y, tree)
        test = Y[z['history'] == 3]
        draw = lambda n, r: test[r.integers(0, len(test), n)]
        extra = {}
    else:
        tree = frozen_null_tree(kind, N, ident)
        Y = null_sample(kind, 200000, N, rng)
        geo = hierarchy_geometry(Y, tree)
        draw = lambda n, r: null_sample(kind, n, N, r)
        extra = {}
    # orthonormal basis of the hierarchy plane
    M = np.vstack([geo['d_coarse'], geo['d_fine']])
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    Q = Vt[s > .2 * s[0]]
    out = dict(tree=tree, geo=geo, draw=draw, Q=Q, **extra)
    _GEOM[key] = out
    return out


def unit_cos(Wm, d):
    return np.abs(Wm @ d) / (np.linalg.norm(Wm, axis=1) * np.linalg.norm(d) + 1e-12)


def mi_binned(x, lab, nlab, bins=16):
    """Miller-Madow corrected plug-in I(lab; x) with equal-mass bins of x (nats)."""
    n = len(x)
    if n < 50: return 0.
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    b = np.searchsorted(edges, x)
    J = np.zeros((bins, nlab)); np.add.at(J, (b, lab), 1)
    pj = J / n; pb = pj.sum(1, keepdims=True); pl = pj.sum(0, keepdims=True)
    nz = pj > 0
    I = float((pj[nz] * np.log(pj[nz] / (pb @ pl)[nz])).sum())
    kb, kl, kj = (pb > 0).sum(), (pl > 0).sum(), nz.sum()
    return max(I - (kj - kb - kl + 1) / (2 * n), 0.)


def cond_mi(x, leaf, bins=16):
    par = leaf // 2; tot = 0.
    for p in (0, 1):
        ix = par == p
        if ix.sum() > 50: tot += ix.mean() * mi_binned(x[ix], leaf[ix] % 2, 2, bins)
    return tot


def probe_info(A, lab, nlab, rng):
    n = len(A); idx = rng.permutation(n); tr, te = idx[: 3 * n // 4], idx[3 * n // 4:]
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
    Z = (A - mu) / sd
    present = np.unique(lab[tr])
    if len(present) < 2: return 0.
    clf = LogisticRegression(max_iter=400, C=1.0).fit(Z[tr], lab[tr])
    P = np.full((len(te), nlab), 1e-12); P[:, clf.classes_] = clf.predict_proba(Z[te]); P /= P.sum(1, keepdims=True)
    ce = -np.log(P[np.arange(len(te)), lab[te]]).mean()
    pri = np.bincount(lab, minlength=nlab) / n
    return float(entropy(pri) - ce)


def np_forward(net, H, g):
    with torch.no_grad():
        m, a = net(torch.as_tensor(H, dtype=torch.float32), torch.full((len(H),), float(np.log(g))), return_hidden=True)
    return m.numpy().astype(float), a.numpy().astype(float)


def jac_sym(net, H, g):
    """Symmetrised analytic Jacobian dm/dh for the one-hidden-layer net, shape (B,N,N)."""
    m, a = np_forward(net, H, g)
    Win = net.inp.weight.detach().numpy().astype(float); V = net.out.weight.detach().numpy().astype(float)
    D = net.D.detach().numpy().astype(float)
    inner = np.einsum('nk,bk,kj->bnj', V, 1 - a ** 2, Win) + np.diag(D)[None]
    J = (1 - m ** 2)[:, :, None] * inner
    return .5 * (J + np.transpose(J, (0, 2, 1)))


def analyze(kind, N, ident, W, seed, full=True):
    t0 = time.monotonic()
    T = target(kind, N, ident); geo = T['geo']; tree = T['tree']; Q = T['Q']
    net = load_net(kind, N, ident, W, seed)
    Win = net.inp.weight.detach().numpy().astype(float)        # (W,N)
    Vout = net.out.weight.detach().numpy().astype(float).T     # (W,N) write directions
    dc = geo['d_coarse']; df = geo['d_fine']
    cos_c = unit_cos(Win, dc); cos_f = np.max([unit_cos(Win, d) for d in df], 0)
    cos_leaf = np.max([unit_cos(Win, mu) for mu in geo['means']], 0)
    frac_plane = np.sum((Win @ Q.T) ** 2, 1) / np.sum(Win ** 2, 1)
    vcos_c = unit_cos(Vout, dc); vcos_f = np.max([unit_cos(Vout, d) for d in df], 0)
    # in-plane angle: coordinates along unit(d_c) and the component of d_f orthogonal to d_c
    ec = dc / np.linalg.norm(dc)
    dfo = df[0] - (df[0] @ ec) * ec; ef = dfo / (np.linalg.norm(dfo) + 1e-12)
    ang = np.degrees(np.arctan2(np.abs(Win @ ef), np.abs(Win @ ec)))
    leaf_angles = [float(np.degrees(np.arctan2(abs(mu @ ef), abs(mu @ ec)))) for mu in geo['means']]
    rng = np.random.default_rng(4040 + seed + 17 * ident + N + W)
    # ---------------- per-gamma unit information, probe information, MSE and ablation
    grid = GU if full else np.array([.06, .28, 1.15])
    unitIc = np.zeros((len(grid), W)); unitIf = np.zeros((len(grid), W)); probe = np.zeros((len(grid), 2))
    mse = np.zeros(len(grid)); abl = np.zeros((len(grid), W))
    for k, g in enumerate(grid):
        Y = T['draw'](NTEST, rng); H = g * Y + np.sqrt(g) * rng.standard_normal(Y.shape)
        _, leaf = predict(Y, tree); coarse = leaf // 2
        m, a = np_forward(net, H, g)
        mse[k] = ((m - Y) ** 2).sum(1).mean()
        for j in range(W):
            unitIc[k, j] = mi_binned(a[:, j], coarse, 2)
            unitIf[k, j] = cond_mi(a[:, j], leaf)
        Ic = probe_info(a, coarse, 2, rng); Il = probe_info(a, leaf, 4, rng)
        probe[k] = [Ic, max(Il - Ic, 0.)]
        if full:
            # mean ablation of each unit
            with torch.no_grad():
                Ht = torch.as_tensor(H, dtype=torch.float32)
                pre = net.inp(Ht) + float(np.log(g)) * net.cg[None, :]
                at = torch.tanh(pre)
                for j in range(W):
                    a2 = at.clone(); a2[:, j] = at[:, j].mean()
                    m2 = torch.tanh(net.out(a2) + net.D[None, :] * Ht).numpy()
                    abl[k, j] = ((m2 - Y) ** 2).sum(1).mean() - mse[k]
    w_log = np.gradient(np.log(grid)); w_log /= w_log.sum()
    importance = (abl * w_log[:, None]).sum(0)

    def half(curve):
        if curve.max() <= 1e-3: return None
        ix = np.flatnonzero(curve >= .5 * curve.max()); return float(grid[ix[0]])
    res = dict(kind=kind, N=N, ident=ident, W=W, seed=seed, plane_dim=int(len(Q)),
               cos_coarse=cos_c, cos_fine=cos_f, cos_leaf=cos_leaf, frac_plane=frac_plane, plane_chance=len(Q) / N,
               write_cos_coarse=vcos_c, write_cos_fine=vcos_f, inplane_angle_deg=ang, leaf_angles_deg=leaf_angles,
               norm_dc=float(np.linalg.norm(dc)), norm_df=np.linalg.norm(df, axis=1), cos_dc_df=[float(abs(d @ dc) / (np.linalg.norm(d) * np.linalg.norm(dc))) for d in df],
               gamma=grid, mse=mse, unit_info_coarse=unitIc, unit_info_fine=unitIf, probe_info=probe,
               unit_half_coarse=[half(unitIc[:, j]) for j in range(W)], unit_half_fine=[half(unitIf[:, j]) for j in range(W)],
               probe_half=[half(probe[:, 0]), half(probe[:, 1])] if full else None,
               ablation=abl, importance=importance, pi=geo['pi'])
    # ---------------- linear response along planted paths (Jacobian spectra)
    if full:
        npth = 160
        Y = T['draw'](npth, rng); _, leaf = predict(Y, tree); par = leaf // 2
        Hp = brownian_fields(Y, GU, rng)
        ec_ = dc / np.linalg.norm(dc); efs = df / np.linalg.norm(df, axis=1, keepdims=True)
        al = np.zeros((len(GU), 2)); lam = np.zeros((len(GU), 2)); ray = np.zeros((len(GU), 2))
        exact_al = np.zeros((len(GU), 2)) if (kind == 'bethe16' and seed == 0) else None
        exact_ray = np.zeros((len(GU), 2)) if (kind == 'bethe16' and seed == 0) else None
        exact_match = np.zeros(len(GU)) if exact_al is not None else None
        for k, g in enumerate(GU):
            J = jac_sym(net, Hp[k], g)
            ev, evec = np.linalg.eigh(J)
            u = evec[:, :, -1]
            al[k] = [np.abs(u @ ec_).mean(), np.mean(np.abs(np.sum(u * efs[par], 1)))]
            lam[k] = [ev[:, -1].mean(), ev[:, -2].mean()]
            top = np.maximum(ev[:, -1], 1e-12)
            ray[k] = [np.mean(np.einsum('i,bij,j->b', ec_, J, ec_) / top),
                      np.mean(np.einsum('bi,bij,bj->b', efs[par], J, efs[par]) / top)]
            if exact_al is not None:
                E = T['E']; ue = []; rr = []
                for i in range(npth):
                    C, _ = E.covariance(Hp[k, i]); w_, v_ = np.linalg.eigh(C); ue.append(v_[:, -1])
                    rr.append([ec_ @ C @ ec_ / max(w_[-1], 1e-12), efs[par[i]] @ C @ efs[par[i]] / max(w_[-1], 1e-12)])
                ue = np.array(ue)
                exact_al[k] = [np.abs(ue @ ec_).mean(), np.mean(np.abs(np.sum(ue * efs[par], 1)))]
                exact_ray[k] = np.mean(rr, 0)
                exact_match[k] = np.mean(np.abs(np.sum(ue * u, 1)))
        res.update(jacobian_alignment=al, jacobian_top_eigs=lam, jacobian_rayleigh=ray, exact_cov_rayleigh=exact_ray, exact_cov_alignment=exact_al, learned_vs_exact_top_eigvec=exact_match,
                   random_unit_vector_baseline=float(np.sqrt(2 / (np.pi * N))))
    res['seconds'] = round(time.monotonic() - t0, 1)
    return res


def analyze_checkpoints(kind, N, ident, W=16, seed=0):
    """Training dynamics: unit alignment and probe information versus optimisation step."""
    T = target(kind, N, ident); geo = T['geo']; tree = T['tree']
    z = np.load(OUT / 'E6B' / 'models' / f'{kind}_N{N}_id{ident}_W{W}_s{seed}_ckpt.npz')
    steps = z['steps']
    rng = np.random.default_rng(5150 + ident + N)
    gsel = (.06, .28, 1.15)
    data = {}
    for g in gsel:
        Y = T['draw'](NTEST, rng); H = g * Y + np.sqrt(g) * rng.standard_normal(Y.shape)
        data[g] = (Y, H, predict(Y, tree)[1])
    rows = []
    for st in list(steps) + [None]:
        net = load_net(kind, N, ident, W, seed, step=None if st is None else int(st))
        Win = net.inp.weight.detach().numpy().astype(float)
        cc = unit_cos(Win, geo['d_coarse']).max(); cf = np.max([unit_cos(Win, d).max() for d in geo['d_fine']])
        row = dict(step=int(st) if st is not None else 'final', best_cos_coarse=float(cc), best_cos_fine=float(cf))
        for g in gsel:
            Y, H, leaf = data[g]
            m, a = np_forward(net, H, g)
            Ic = probe_info(a, leaf // 2, 2, np.random.default_rng(1)); Il = probe_info(a, leaf, 4, np.random.default_rng(1))
            row[f'probe_coarse@{g}'] = Ic; row[f'probe_fine@{g}'] = max(Il - Ic, 0.)
            row[f'mse@{g}'] = float(((m - Y) ** 2).sum(1).mean())
        rows.append(row)
    return rows


def run_all(jobs, ckpt_targets):
    """Full analysis for W=16; geometry + three-gamma probes for the capacity sweep."""
    d = OUT / 'E6B' / 'analysis'; d.mkdir(parents=True, exist_ok=True)
    for kind, N, ident, W, seed in jobs:
        f = d / f'{kind}_N{N}_id{ident}_W{W}_s{seed}.json'
        if f.exists(): continue
        try:
            r = analyze(kind, N, ident, W, seed, full=(W == 16))
        except FileNotFoundError as e:
            print('MISSING', e); continue
        json_dump(f, r)
        print('ANALYZED', f.stem, 'best cos c/f', round(float(np.max(r['cos_coarse'])), 3), round(float(np.max(r['cos_fine'])), 3),
              'probe', np.round(np.array(r['probe_info']).max(0), 3), 's', r['seconds'], flush=True)
    for kind, N, ident in ckpt_targets:
        f = d / f'ckpt_{kind}_N{N}_id{ident}.json'
        if f.exists(): continue
        try:
            rows = analyze_checkpoints(kind, N, ident)
        except FileNotFoundError as e:
            print('MISSING', e); continue
        json_dump(f, rows); print('CKPT', f.stem, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--jobs', required=True); p.add_argument('--ckpt', default=None)
    a = p.parse_args()
    jobs = [tuple(j[:5]) for j in json.loads(Path(a.jobs).read_text())]
    ck = [tuple(j) for j in json.loads(Path(a.ckpt).read_text())] if a.ckpt else []
    run_all(jobs, ck)
