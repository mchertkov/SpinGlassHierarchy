"""Stage 6B summary + figure: what a learned PID denoiser represents.

Reads results/E6/E6B/analysis/*.json and results/E6/reference_curves.json.
Writes results/E6/STAGE6B_SUMMARY.json and results/E6/figures/E6B_learned_units.{pdf,png}.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from e6_common import ROOT, OUT, json_dump, N16_ACCEPTED, N16_REJECTED

CO, FI, GR = '#2864AB', '#D2691E', '#7A7A7A'
A = OUT / 'E6B' / 'analysis'
FIG = ROOT / 'results' / 'E6' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
REF = json.loads((OUT / 'reference_curves.json').read_text())
BETHE = [('bethe16', 16, s) for s in N16_ACCEPTED] + [('bethe', N, d) for N in (32, 64) for d in (4, 5, 8)]
REJ = [('bethe16', 16, s) for s in N16_REJECTED]
NULLS = [(m, N, 4) for m in ('two_state', 'product') for N in (32, 64)]


def get(kind, N, ident, W=16, seed=0):
    f = A / f'{kind}_N{N}_id{ident}_W{W}_s{seed}.json'
    return json.loads(f.read_text()) if f.exists() else None


def ref_half(kind, N, ident):
    r = REF.get(f'{kind}_N{N}_id{ident}')
    if r is None: return [None, None]
    g = np.array(r['gamma']); I = np.array(r['info']); k = np.searchsorted(g, 3.2 - 1e-12)
    out = []
    for l in range(2):
        ix = np.flatnonzero(I[:k + 1, l] >= .5 * I[k, l]); out.append(float(g[ix[0]]) if len(ix) and I[k, l] > 0.02 else None)
    return out


def unit_summary(r, thr=.7):
    cc, cf = np.array(r['cos_coarse']), np.array(r['cos_fine']); imp = np.array(r['importance'])
    frac = np.array(r['frac_plane'])
    Ic = np.array(r['unit_info_coarse']).max(0); If = np.array(r['unit_info_fine']).max(0)
    jc, jf = int(np.argmax(Ic)), int(np.argmax(If))
    return dict(n_coarse_aligned=int((cc >= thr).sum()), n_fine_aligned=int((cf >= thr).sum()),
                best_cos_coarse=float(cc.max()), best_cos_fine=float(cf.max()),
                frac_plane_median=float(np.median(frac)), frac_plane_importance_weighted=float((frac * np.maximum(imp, 0)).sum() / max(np.maximum(imp, 0).sum(), 1e-12)),
                plane_chance=r['plane_chance'],
                coarse_unit=dict(index=jc, cos_coarse=float(cc[jc]), cos_fine=float(cf[jc]), info=float(Ic[jc]), half=r['unit_half_coarse'][jc]),
                fine_unit=dict(index=jf, cos_coarse=float(cc[jf]), cos_fine=float(cf[jf]), info=float(If[jf]), half=r['unit_half_fine'][jf]),
                probe_max=np.array(r['probe_info']).max(0).tolist(), probe_half=r['probe_half'])


def main():
    S = dict(targets={})
    for grp, lst in (('bethe', BETHE), ('rejected16', REJ), ('nulls', NULLS)):
        for kind, N, ident in lst:
            rs = [get(kind, N, ident, 16, s) for s in (0, 1, 2)]
            rs = [r for r in rs if r is not None]
            if not rs: continue
            key = f'{kind}_N{N}_id{ident}'
            S['targets'][key] = dict(group=grp, N=N, ref_half=ref_half(kind, N, ident), seeds=[unit_summary(r) for r in rs])
            r0 = rs[0]
            if r0.get('jacobian_alignment') is not None:
                S['targets'][key]['jacobian_rayleigh'] = r0['jacobian_rayleigh']
                S['targets'][key]['jacobian_alignment'] = r0['jacobian_alignment']
                if r0.get('exact_cov_rayleigh') is not None:
                    S['targets'][key]['exact_cov_rayleigh'] = r0['exact_cov_rayleigh']
                    S['targets'][key]['learned_vs_exact_top_eigvec'] = r0['learned_vs_exact_top_eigvec']
            # cross-seed reproducibility of the coarse and fine units' input weights
            try:
                import e6b_train as T          # needs PyTorch; the cross-seed statistic is skipped without it (figure unaffected)
            except ImportError:
                T = None
            if len(rs) >= 2 and T is not None:
                ws = [T.load_net(kind, N, ident, 16, s).inp.weight.detach().numpy().astype(float) for s in range(len(rs))]
                rep = {}
                for lev, keyu in (('coarse', 'coarse_unit'), ('fine', 'fine_unit')):
                    idx = [S['targets'][key]['seeds'][s][keyu]['index'] for s in range(len(rs))]
                    vs = [ws[s][idx[s]] / np.linalg.norm(ws[s][idx[s]]) for s in range(len(rs))]
                    rep[lev] = [float(abs(vs[i] @ vs[j])) for i in range(len(vs)) for j in range(i + 1, len(vs))]
                S['targets'][key]['cross_seed_unit_cos'] = rep
    # group statistics
    def coll(grp, f):
        return [f(t) for t in S['targets'].values() if t['group'] == grp]
    for grp in ('bethe', 'rejected16', 'nulls'):
        S[f'{grp}_stats'] = dict(
            best_cos_coarse_median=float(np.median(coll(grp, lambda t: np.median([s['best_cos_coarse'] for s in t['seeds']])))),
            best_cos_fine_median=float(np.median(coll(grp, lambda t: np.median([s['best_cos_fine'] for s in t['seeds']])))),
            probe_fine_max_median=float(np.median(coll(grp, lambda t: np.median([s['probe_max'][1] for s in t['seeds']])))),
            probe_coarse_max_median=float(np.median(coll(grp, lambda t: np.median([s['probe_max'][0] for s in t['seeds']])))),
            frac_plane_weighted_median=float(np.median(coll(grp, lambda t: np.median([s['frac_plane_importance_weighted'] for s in t['seeds']])))),
            plane_chance_median=float(np.median(coll(grp, lambda t: t['seeds'][0]['plane_chance']))),
            n_fine_aligned_median=float(np.median(coll(grp, lambda t: np.median([s['n_fine_aligned'] for s in t['seeds']])))),
            n_coarse_aligned_median=float(np.median(coll(grp, lambda t: np.median([s['n_coarse_aligned'] for s in t['seeds']])))),
            cross_seed_cos_coarse_median=float(np.median(sum(coll(grp, lambda t: t.get('cross_seed_unit_cos', {}).get('coarse', [])), []) or [np.nan])),
            cross_seed_cos_fine_median=float(np.median(sum(coll(grp, lambda t: t.get('cross_seed_unit_cos', {}).get('fine', [])), []) or [np.nan])),
            n_targets=len(coll(grp, lambda t: 1)))
    # unit half-times vs class half-times (Bethe, seed 0)
    pairs = []
    for key, t in S['targets'].items():
        if t['group'] != 'bethe': continue
        rh = t['ref_half']; s0 = t['seeds'][0]
        pairs.append(dict(key=key, N=t['N'], ref=rh, unit=[s0['coarse_unit']['half'], s0['fine_unit']['half']], probe=s0['probe_half']))
    S['unit_vs_class_half'] = pairs
    ok = [p for p in pairs if None not in p['ref'] and None not in p['unit']]
    S['unit_order'] = dict(n=len(ok), coarse_unit_before_fine_unit=int(sum(p['unit'][0] < p['unit'][1] for p in ok)),
                           ties=int(sum(p['unit'][0] == p['unit'][1] for p in ok)),
                           median_abs_log_ratio_coarse=float(np.median([abs(np.log(p['unit'][0] / p['ref'][0])) for p in ok])),
                           median_abs_log_ratio_fine=float(np.median([abs(np.log(p['unit'][1] / p['ref'][1])) for p in ok])))
    # training dynamics and capacity
    ck = {}
    for kind, N, ident in BETHE + REJ + NULLS:
        f = A / f'ckpt_{kind}_N{N}_id{ident}.json'
        if f.exists(): ck[f'{kind}_N{N}_id{ident}'] = json.loads(f.read_text())
    S['training_order'] = {}
    def t_reach(rows, key, frac=.9):
        v = np.array([r[key] for r in rows[:-1]]); st = np.array([r['step'] for r in rows[:-1]])
        ix = np.flatnonzero(v >= frac * v[-1]); return int(st[ix[0]])
    for k, rows in ck.items():
        S['training_order'][k] = dict(coarse=t_reach(rows, 'best_cos_coarse'), fine=t_reach(rows, 'best_cos_fine'),
                                      init_cos=[rows[0]['best_cos_coarse'], rows[0]['best_cos_fine']],
                                      final_cos=[rows[-1]['best_cos_coarse'], rows[-1]['best_cos_fine']],
                                      peak_cos=[max(r['best_cos_coarse'] for r in rows), max(r['best_cos_fine'] for r in rows)])
    bt = [v for k, v in S['training_order'].items() if any(k == f'{a}_N{b}_id{c}' for a, b, c in BETHE)]
    S['training_order_stats'] = dict(n=len(bt), coarse_first=int(sum(v['coarse'] < v['fine'] for v in bt)), ties=int(sum(v['coarse'] == v['fine'] for v in bt)),
                                     median_ratio_fine_over_coarse=float(np.median([v['fine'] / v['coarse'] for v in bt])),
                                     median_init_cos=np.median([v['init_cos'] for v in bt], 0).tolist(),
                                     median_final_cos=np.median([v['final_cos'] for v in bt], 0).tolist(),
                                     note='step at which the best unit-contrast cosine first reaches 90% of its final value; init = 10-step checkpoint')
    cap = {}
    for kind, N, ident in BETHE + NULLS:
        row = {}
        for W in (1, 2, 3, 4, 8, 16, 32):
            r = get(kind, N, ident, W, 0)
            if r is None: continue
            P = np.array(r['probe_info']); g = list(np.round(r['gamma'], 4))
            def at(gg, l):
                return float(P[int(np.argmin(np.abs(np.array(r['gamma']) - gg))), l])
            Uc = np.array(r['unit_info_coarse']); Uf = np.array(r['unit_info_fine']); gg_ = np.array(r['gamma'])
            row[W] = dict(coarse=at(.06, 0), fine=max(at(.28, 1), at(1.15, 1)), best_cos_coarse=float(max(r['cos_coarse'])), best_cos_fine=float(max(r['cos_fine'])),
                          unit_coarse=float(Uc[int(np.argmin(np.abs(gg_ - .06)))].max()), unit_fine=float(max(Uf[int(np.argmin(np.abs(gg_ - .28)))].max(), Uf[int(np.argmin(np.abs(gg_ - 1.15)))].max())))
        cap[f'{kind}_N{N}_id{ident}'] = row
    S['capacity'] = cap
    json_dump(OUT / 'STAGE6B_SUMMARY.json', S)
    # ================================================================ figure
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.2))
    mk = {16: 'o', 32: 's', 64: 'D'}
    # (a)/(b) unit directions in the hierarchy plane
    for a, grp, title in ((ax[0, 0], BETHE, '(a) Units of Bethe denoisers (gate passed)'), (ax[0, 1], REJ + NULLS, '(b) Controls: failed-gate N=16, two-state, product')):
        for kind, N, ident in grp:
            for s in (0, 1, 2):
                r = get(kind, N, ident, 16, s)
                if r is None: continue
                ang = np.radians(np.array(r['inplane_angle_deg'])); rad = np.sqrt(np.array(r['frac_plane']))
                imp = np.maximum(np.array(r['importance']), 0); size = 6 + 120 * imp / max(imp.max(), 1e-9)
                col = {'bethe16': '#444444', 'bethe': '#444444', 'two_state': FI, 'product': '#2E8B57'}[kind]
                if grp is not BETHE and kind == 'bethe16': col = '#9B59B6'
                a.scatter(rad * np.cos(ang), rad * np.sin(ang), s=size, marker=mk[N], facecolor='none', edgecolor=col, lw=.6, alpha=.7)
        th = np.linspace(0, np.pi / 2, 100); a.plot(np.cos(th), np.sin(th), color='k', lw=.6)
        a.set_xlim(0, 1.02); a.set_ylim(0, 1.02); a.set_aspect('equal')
        a.set_xlabel(r'$|\hat w\cdot\hat d_{\rm coarse}|$'); a.set_ylabel(r'$|\hat w\cdot\hat e_{\rm fine}|$ ($\perp$ part of $d_{\rm fine}$)')
        a.set_title(title, fontsize=9.5, loc='left')
    ax[0, 0].legend(handles=[Line2D([], [], marker=m, ls='', mfc='none', mec='#444444', label=f'N={n}') for n, m in mk.items()] +
                    [Line2D([], [], ls='', label='size $\\propto$ ablation importance')], fontsize=7, frameon=False, loc='upper right')
    ax[0, 1].legend(handles=[Line2D([], [], marker='o', ls='', mfc='none', mec='#9B59B6', label='Bethe N=16, failed gate'),
                             Line2D([], [], marker='s', ls='', mfc='none', mec=FI, label='two-state (N=32,64)'),
                             Line2D([], [], marker='s', ls='', mfc='none', mec='#2E8B57', label='independent spins (N=32,64)')], fontsize=7, frameon=False, loc='upper right')
    # (c) unit vs class half-times
    c = ax[0, 2]
    for p in pairs:
        for l, col in ((0, CO), (1, FI)):
            if p['ref'][l] and p['unit'][l]:
                c.scatter(p['ref'][l], p['unit'][l], marker=mk[p['N']], s=36, facecolor=col, edgecolor=col, alpha=.85)
    lim = [0.02, 4]; c.plot(lim, lim, ':', color='k', lw=.8); c.set_xscale('log'); c.set_yscale('log'); c.set_xlim(lim); c.set_ylim(lim)
    c.set_xlabel(r'class half-information $\gamma^{1/2}$ (exact or sampled posterior)'); c.set_ylabel(r'single-unit half-information $\gamma^{1/2}$')
    c.set_title('(c) Unit decision windows match class windows', fontsize=9.5, loc='left')
    c.legend(handles=[Line2D([], [], marker='o', ls='', color=CO, label='coarse unit'), Line2D([], [], marker='o', ls='', color=FI, label='fine unit')], fontsize=7, frameon=False, loc='upper left')
    # (d) linear response: Rayleigh ratios along contrasts
    d = ax[1, 0]
    GU = np.array([.005, .01, .02, .04, .06, .085, .12, .16, .21, .28, .36, .46, .58, .72, .9, 1.15, 1.45, 1.85, 2.4, 3.2])  # grid of e6b_analyze.py
    def stack(keys, field):
        v = [np.array(S['targets'][k][field]) for k in keys if k in S['targets'] and S['targets'][k].get(field) is not None]
        return np.array(v) if v else None
    k16 = [f'bethe16_N16_id{s}' for s in N16_ACCEPTED]; kL = [f'bethe_N{N}_id{dd}' for N in (32, 64) for dd in (4, 5, 8)]
    ex = stack(k16, 'exact_cov_rayleigh'); l16 = stack(k16, 'jacobian_rayleigh'); lL = stack(kL, 'jacobian_rayleigh')
    ts = stack(['two_state_N32_id4', 'two_state_N64_id4'], 'jacobian_rayleigh')
    for arr, ls, lab in ((ex, '--', 'exact covariance, N=16'), (l16, '-', 'learned Jacobian, N=16'), (lL, '-.', 'learned Jacobian, N=32/64')):
        if arr is None: continue
        d.plot(GU, np.median(arr[:, :, 0], 0), ls, color=CO, lw=1.8); d.plot(GU, np.median(arr[:, :, 1], 0), ls, color=FI, lw=1.8)
    if ts is not None: d.plot(GU, np.median(ts[:, :, 1], 0), ':', color=GR, lw=1.5)
    d.set_xscale('log'); d.set_ylim(0, 1.02); d.set_xlabel(r'$\gamma$'); d.set_ylabel(r'$\hat d^{\top} C_\gamma \hat d\,/\,\lambda_{\max}(C_\gamma)$')
    d.set_title('(d) Linear response along the contrasts', fontsize=9.5, loc='left')
    d.legend(handles=[Line2D([], [], color='k', ls='--', label='exact covariance, N=16'), Line2D([], [], color='k', ls='-', label='learned Jacobian, N=16'),
                      Line2D([], [], color='k', ls='-.', label='learned Jacobian, N=32/64'), Line2D([], [], color=GR, ls=':', label='two-state, fine direction'),
                      Line2D([], [], color=CO, label=r'$\hat d=\hat d_{\rm coarse}$'), Line2D([], [], color=FI, label=r'$\hat d=\hat d_{\rm fine}$ (planted parent)')], fontsize=6.5, frameon=False, loc='center left', bbox_to_anchor=(0.0, 0.66))
    # (e) training dynamics
    e = ax[1, 1]
    for lev, keyc, col in ((0, 'best_cos_coarse', CO), (1, 'best_cos_fine', FI)):
        curves = []
        for k in [f'{a}_N{b}_id{cc}' for a, b, cc in BETHE]:
            rows = ck.get(k)
            if not rows: continue
            st = np.array([r['step'] for r in rows[:-1]]); v = np.array([r[keyc] for r in rows[:-1]])
            e.plot(st, v, color=col, lw=.6, alpha=.35); curves.append(v)
        if curves: e.plot(st, np.median(curves, 0), color=col, lw=2.2)
    e.set_xscale('log'); e.set_ylim(0, 1.02); e.set_xlabel('optimisation step'); e.set_ylabel('best unit |cos| with contrast')
    e.set_title('(e) Training: coarse contrast is learned first', fontsize=9.5, loc='left')
    e.legend(handles=[Line2D([], [], color=CO, lw=2, label=r'$d_{\rm coarse}$'), Line2D([], [], color=FI, lw=2, label=r'$d_{\rm fine}$')], fontsize=7, frameon=False, loc='lower right')
    # (f) capacity
    f_ = ax[1, 2]
    Ws = [1, 2, 3, 4, 8, 16, 32]
    def capcurve(keys, field):
        M = []
        for k in keys:
            row = cap.get(k, {})
            M.append([row[W][field] if W in row else np.nan for W in Ws])
        return np.array(M)
    kb = [f'{a}_N{b}_id{cc}' for a, b, cc in BETHE]
    for field, col in (('unit_coarse', CO), ('unit_fine', FI)):
        M = capcurve(kb, field)
        if np.isfinite(M).any():
            f_.plot(Ws, np.nanmedian(M, 0), 'o-', color=col, lw=1.8)
            f_.fill_between(Ws, np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0), color=col, alpha=.15, lw=0)
    Mt = capcurve(['two_state_N32_id4', 'two_state_N64_id4'], 'unit_fine')
    if np.isfinite(Mt).any(): f_.plot(Ws, np.nanmedian(Mt, 0), 's:', color=GR, lw=1.3)
    f_.set_xscale('log', base=2); f_.set_xticks(Ws); f_.set_xticklabels(Ws); f_.set_xlabel('hidden width W'); f_.set_ylabel('best single-unit information (nats)')
    f_.set_title('(f) Capacity: a coarse unit first, fine units later', fontsize=9.5, loc='left')
    f_.legend(handles=[Line2D([], [], color=CO, marker='o', label=r'coarse label ($\gamma$=0.06)'), Line2D([], [], color=FI, marker='o', label=r'fine | coarse ($\gamma$=0.28 or 1.15)'),
                       Line2D([], [], color=GR, ls=':', marker='s', label='two-state, fine')], fontsize=7, frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(FIG / f'E6B_learned_units.{ext}', dpi=200)
    print(json.dumps({k: v for k, v in S.items() if k not in ('targets', 'capacity', 'training_order', 'unit_vs_class_half')}, indent=1))
    print('capacity medians (Bethe):', {W: [float(np.nanmedian(capcurve(kb, f)[:, i])) for f in ('unit_coarse', 'unit_fine', 'best_cos_coarse', 'best_cos_fine')] for i, W in enumerate(Ws)})


if __name__ == '__main__':
    main()
