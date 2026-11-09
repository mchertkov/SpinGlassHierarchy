"""New campaign N1 (2026-10-31): exact level-resolved decisions with an explicit Z2 level and a
non-spectral, data-depth hierarchy (code/exacttree.py).

usage:  python experiments/n1_levels.py scan  --family bethe --N 16 --betas 0.5 1 1.5 2 2.5 --seeds 0-59
        python experiments/n1_levels.py anatomy --family bethe --N 24 --beta 2.5 --seed 2
        python experiments/n1_levels.py hopfield
Outputs: results/N1/<family>_N<N>_b<beta>_s<seed>.json  (+ .npz for anatomy runs)
"""
from pathlib import Path
import sys, json, time, argparse
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import exacttree as E

OUT = ROOT / 'results' / 'N1'; OUT.mkdir(parents=True, exist_ok=True)


def gammas_for(N, n=24, tau_min=0.02, gmax=5.0):
    return np.concatenate([[0.0], np.geomspace(tau_min / N, gmax, n)])


def analyze(J, name, N, npaths, seed, cov_paths=0, save_npz=False, tree_kw=None):
    tic = time.time(); rng = np.random.default_rng(7_000_003 + seed + 1000 * int(__import__('os').environ.get('N1_REP', '0')))
    M = E.ExactModel(J, name); T = E.build_tree(M, **(tree_kw or {}))
    g = gammas_for(N)
    R = E.run_paths(M, T, g, npaths, rng, cov_paths=cov_paths)
    I, dI, Hl, dH = E.level_information(M, T, R)
    L = len(T['labels'])
    frac = np.array([dI[l] / dH[l] if dH[l] > 1e-9 else np.full(len(g), np.nan) for l in range(L)])
    th = [E.half_time(g, frac[l]) if dH[l] > 1e-9 else None for l in range(L)]
    lam_cond = E.conditional_lambdas(M, T); chans = E.parent_channels(M, T)
    # bootstrap of half-times over planted paths (resampling within strata is approximated by weighted resampling)
    boot = []
    brng = np.random.default_rng(seed + 99); n = len(R['w'])
    for _ in range(60):
        idx = brng.integers(0, n, n); Rb = dict(R); Rb['post'] = [p[idx] for p in R['post']]; wb = R['w'][idx]; Rb['w'] = wb / wb.sum()
        Ib, dIb, _, _ = E.level_information(M, T, Rb)
        boot.append([E.half_time(g, dIb[l] / dH[l]) if dH[l] > 1e-9 else np.nan for l in range(len(T['labels']))])
    boot = np.array(boot, float)
    out = dict(name=name, N=N, npaths=npaths, gamma=g.tolist(), n_levels_glassy=L - 1,
               n_classes=T['n_classes'], level_info=T['level_info'], candidates=T['candidates'], channels=chans, valley_mass=T['valley_mass'].tolist(),
               lam=M.lam.tolist(), lam_cond=lam_cond.tolist(), H_levels=Hl.tolist(), dH=dH.tolist(),
               info=I.tolist(), dinfo=dI.tolist(), frac=frac.tolist(), half_gamma=th,
               pred_cond=[float(E.S_STAR / x) for x in lam_cond],
               half_gamma_ci=[[float(np.nanquantile(boot[:, l], .16)), float(np.nanquantile(boot[:, l], .84))] for l in range(L)],
               pred_seq=[float(E.S_STAR / M.lam[l]) for l in range(L)],
               qbar=(R['w'][:, None] * R['qself']).sum(0).tolist(), npaths_eff=int(len(R['w'])), chiSG0_over_N=float((M.C0 ** 2).sum() / N ** 2),
               seconds=time.time() - tic)
    # per-path commitment / recrossing at each level (planted class mass >= .9, later < .8)
    comm = []
    npaths = len(R['Y'])
    for l in range(L):
        pp = R['post'][l][np.arange(npaths), :, R['plant'][l]]
        hit = (pp >= 0.9); first = np.where(hit.any(1), hit.argmax(1), -1); w = R['w']
        rec = np.array([bool(f >= 0 and (pp[i, f:] < 0.8).any()) for i, f in enumerate(first)])
        hg = g[np.maximum(first, 0)]; ok = first >= 0
        med = None
        if ok.any():
            o_ = np.argsort(hg[ok]); cw = np.cumsum(w[ok][o_]) / w[ok].sum(); med = float(hg[ok][o_][np.searchsorted(cw, 0.5)])
        comm.append(dict(hit_frac=float(w[ok].sum()), recross_frac=float(w[rec].sum()), median_hit_gamma=med))
    out['commitment'] = comm
    if cov_paths:
        # |cos| between the top posterior-covariance direction and the planted contrasts per level
        cosines = np.zeros((cov_paths, len(g), L))
        for i in range(cov_paths):
            for l in range(L):
                if l == 0:
                    Wp = M.p * (T['labels'][0] == T['labels'][0].max()); d = M.first_moment(Wp) / Wp.sum()
                else:
                    pc = R['plant'][l - 1][i]
                    kids = np.unique(T['labels'][l][T['labels'][l - 1] == pc])
                    if len(kids) < 2: cosines[i, :, l] = np.nan; continue
                    d = E.contrasts(M, T, l, pc)
                d = d / np.linalg.norm(d)
                cosines[i, :, l] = np.abs(R['topvec'][i] @ d)
        out['top_cos_median'] = np.nanmedian(cosines, 0).tolist()
        out['top_lam_median'] = np.median(R['toplam'], 0).tolist()
    if save_npz:
        np.savez_compressed(OUT / f'{name}.npz', gamma=g, Y=R['Y'], w=R['w'], qself=R['qself'],
                            **{f'post{l}': R['post'][l] for l in range(L)},
                            **{f'plant{l}': R['plant'][l] for l in range(L)},
                            **({'topcos': cosines, 'toplam': R['toplam']} if cov_paths else {}))
    json.dump(out, open(OUT / f'{name}.json', 'w'), indent=1)
    return out


xs_ = np.linspace(-8, 8, 2001); wx_ = np.exp(-xs_ ** 2 / 2); wx_ /= wx_.sum()


def pred_half(chans):
    """Half-information time of the binary-channel mixture sum_P w_P I_bin(gamma kappa_P), w_P ~ pi_P H_P."""
    w = np.array([c['pi'] * c['H'] for c in chans]); w = w / w.sum(); gg = np.geomspace(1e-4, 50, 1200)
    u = gg[:, None, None] * np.array([c['kappa'] for c in chans])[None, :, None] + np.sqrt(gg[:, None, None] * np.array([c['kappa'] for c in chans])[None, :, None]) * xs_
    Ib = (np.log(2) - (np.logaddexp(0, -2 * u) * wx_).sum(-1)) / np.log(2)
    return float(gg[np.argmax((Ib * w).sum(1) >= 0.5)])


def structure(J, name, N):
    """Classes only (no planted paths): tree, collective strengths, spectra, binarity, predicted decision times,
    and the unit contrast of every accepted split (gauge fixed by the ground state, which does not depend on beta)."""
    tic = time.time(); M = E.ExactModel(J, name); T = E.build_tree(M); chans = E.parent_channels(M, T)
    v = np.linalg.eigh(M.C0)[1][:, -1]; binar = float(M.lam[0] / (M.p * np.abs(M.dot_grid(v))).sum() ** 2)
    contr = []
    for l in range(1, len(T['labels'])):
        par = T['labels'][l - 1]; K = T['n_classes'][l - 1]; row = []
        for c in range(K // 2, K):                                   # parents in the + sector
            kids = np.unique(T['labels'][l][par == c])
            if len(kids) == 2:
                d = E.contrasts(M, T, l, c); row.append(dict(parent=int(c), d=(d / np.linalg.norm(d)).tolist()))
        contr.append(row)
    out = dict(name=name, N=N, n_levels_glassy=len(T['labels']) - 1, n_classes=T['n_classes'], level_info=T['level_info'],
               candidates=T['candidates'], channels=chans, lam=M.lam[:8].tolist(), binarity=binar, ground_state=M.g.tolist(),
               pred_half=[pred_half(ch) for ch in chans], contrasts=contr, chiSG0_over_N=float((M.C0 ** 2).sum() / N ** 2),
               valley_mass=T['valley_mass'].tolist(), seconds=time.time() - tic)
    (ROOT / 'results' / 'N1S').mkdir(exist_ok=True)
    json.dump(out, open(ROOT / 'results' / 'N1S' / f'{name}.json', 'w'), indent=1)
    return out


def J_of(family, N, beta, seed):
    if family == 'bethe': return E.bethe_J(N, beta, seed)
    if family == 'ferro': return np.abs(E.bethe_J(N, beta, seed))          # same graph and |J|, no frustration
    if family == 'sk': return E.sk_J(N, beta, seed)
    if family == 'mattis':                                                   # two-state null, m ~ 0.8 at beta=1.37
        xi = np.random.default_rng(seed).choice([-1.0, 1.0], N); return beta / N * np.outer(xi, xi)
    raise ValueError(family)


def parse_seeds(s):
    if '-' in s: a, b = s.split('-'); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode')
    ap.add_argument('--family', default='bethe'); ap.add_argument('--N', type=int, default=16)
    ap.add_argument('--betas', type=float, nargs='*', default=[2.5]); ap.add_argument('--beta', type=float, default=2.5)
    ap.add_argument('--seeds', default='1-12'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--paths', type=int, default=48); ap.add_argument('--cov', type=int, default=0)
    a = ap.parse_args()
    if a.mode == 'scan':
        for b in a.betas:
            for s in parse_seeds(a.seeds):
                nm = f'{a.family}_N{a.N}_b{b}_s{s}'
                if (OUT / f'{nm}.json').exists(): continue
                o = analyze(J_of(a.family, a.N, b, s), nm, a.N, a.paths, s)
                print(nm, 'levels', o['n_levels_glassy'], 'half', [None if x is None else round(x, 4) for x in o['half_gamma']],
                      'pred', [round(x, 4) for x in o['pred_cond']], '%.1fs' % o['seconds'], flush=True)
    elif a.mode == 'structure':
        for b in a.betas:
            for s in parse_seeds(a.seeds):
                nm = f'{a.family}_N{a.N}_b{b}_s{s}'
                if (ROOT / 'results' / 'N1S' / f'{nm}.json').exists(): continue
                o = structure(J_of(a.family, a.N, b, s), nm, a.N)
                print('S', nm, 'levels', o['n_levels_glassy'], 'kappa1', [round(c['kappa'], 2) for c in o['candidates'] if c['level'] == 1],
                      'pred', [round(x, 3) for x in o['pred_half']], '%.1fs' % o['seconds'], flush=True)
    elif a.mode == 'anatomy':
        nm = f'{a.family}_N{a.N}_b{a.beta}_s{a.seed}_anatomy'
        o = analyze(J_of(a.family, a.N, a.beta, a.seed), nm, a.N, a.paths, a.seed, cov_paths=a.cov, save_npz=True)
        print(json.dumps({k: o[k] for k in ['n_classes', 'half_gamma', 'pred_cond', 'pred_seq', 'commitment', 'seconds']}, indent=1))
