"""Stage 6A summary + figure: algorithmic uses of the contrast identity.

Outputs results/E6/STAGE6A_SUMMARY.json and results/E6/figures/E6A_contrast_algorithms.{pdf,png}.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e6_common import ROOT, OUT, json_dump, N16_ACCEPTED, N16_REJECTED
from e6a_contrast_algorithms import cumtrapz0

CO, FI, GR = '#2864AB', '#D2691E', '#7A7A7A'
FIG = ROOT / 'results' / 'E6' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)


def load(model, s):
    return json.loads((OUT / 'E6A' / f'{model}_seed{s}.json').read_text())


def rate_rmse(model, s, n=24, reps=300):
    z = np.load(OUT / 'E6A' / f'{model}_seed{s}.npz'); g = z['grid']
    dC, dF, rc, rf = (z[k].astype(float) for k in ('dropC', 'dropF', 'rc', 'rf'))
    true = np.stack([rc.mean(1), rf.mean(1)], 1)                      # 1536-path contrast rate
    rng = np.random.default_rng(77 + s)
    gm = np.sqrt(g[1:] * g[:-1])                                         # FD rates live at midpoints
    true_mid = np.stack([np.interp(np.log(gm), np.log(g[1:]), true[1:, l]) for l in range(2)], 1)
    eF, eC = [], []
    for _ in range(reps):
        ii = rng.integers(0, dC.shape[1], n)
        ie = np.stack([dC[:, ii].mean(1), dF[:, ii].mean(1)], 1)
        fd = np.diff(ie, axis=0) / np.diff(g)[:, None]
        rcn = np.stack([rc[:, ii].mean(1), rf[:, ii].mean(1)], 1)
        rc_mid = np.stack([np.interp(np.log(gm), np.log(g[1:]), rcn[1:, l]) for l in range(2)], 1)
        eF.append(fd - true_mid); eC.append(rc_mid - true_mid)
    eF, eC = np.array(eF), np.array(eC)
    return gm, np.sqrt((eF ** 2).mean(0)), np.sqrt((eC ** 2).mean(0)), true_mid


def _gridsteps(rows):
    from e5b_sequential_smc import GRID as G5
    out = {}
    for model in ('bethe', 'two_state', 'product'):
        st = [abs(int(np.searchsorted(G5, x['predicted_grid'][l] - 1e-9)) - int(np.searchsorted(G5, x['measured'][l] - 1e-9)))
              for x in rows if x['N'] > 16 and x['model'] == model for l in (0, 1)]
        out[model] = dict(n=len(st), exact=int(sum(v == 0 for v in st)), within_one=int(sum(v <= 1 for v in st)), max_steps=int(max(st)))
    return out


def main():
    S = {}
    acc = [load('bethe', s) for s in N16_ACCEPTED]
    allb = [load('bethe', s) for s in N16_ACCEPTED + N16_REJECTED]
    nulls = [load(m, s) for m in ('two_state', 'product') for s in (4, 5, 6, 7, 8)]
    g = np.array(acc[0]['grid'])
    # ---- identity check (exclude gamma=0 where both sides vanish)
    z = []
    for r in allb + nulls:
        zz = np.load(OUT / 'E6A' / f"{r['model']}_seed{r['seed']}.npz")
        for d, rr in ((zz['dropC'], zz['rc']), (zz['dropF'], zz['rf'])):
            M = d.astype(float) - cumtrapz0(rr.astype(float), g)
            se = M.std(1) / np.sqrt(M.shape[1])
            z.append(np.max(np.abs(M.mean(1)[1:]) / np.maximum(se[1:], 1e-12)))
    S['identity'] = dict(max_abs_gap_nats=max(r['identity_max_abs_gap'] for r in allb + nulls),
                         max_abs_z_over_gamma_gt0=float(np.max(z)), median_max_abs_z=float(np.median(z)),
                         note='gap = mean entropy drop minus integrated mean contrast rate, 1536 exact paths; z in units of path standard error')
    # ---- budget (accepted N16)
    B = {k: np.array([r['budget'][k] for r in acc]) for k in ('total', 'coarse', 'fine', 'within')}
    frac = {k: B[k] / B['total'] for k in ('coarse', 'fine', 'within')}
    S['budget'] = dict(identity_max_abs=max(r['budget_identity_max_abs'] for r in acc),
                       hierarchy_info_at_5=[r['hierarchy_information_at_5'] for r in acc],
                       total_info_at_5=[r['total_information_at_5'] for r in acc],
                       gibbs_entropy=[r['gibbs_entropy'] for r in acc],
                       peak_hierarchy_fraction_median=float(np.median((frac['coarse'] + frac['fine']).max(1))),
                       hierarchy_fraction_at_gamma0_median=float(np.median(frac['coarse'][:, 0] + frac['fine'][:, 0])))
    for gg in (0.05, 0.1, 0.3, 1.0, 3.0):
        k = int(np.argmin(np.abs(g - gg)))
        S['budget'][f'fractions_at_{g[k]:.3g}'] = {kk: float(np.median(v[:, k])) for kk, v in frac.items()}
    # ---- rate RMSE
    rr = [rate_rmse('bethe', s) for s in N16_ACCEPTED]
    gm = rr[0][0]
    ratio = np.array([x[1] / np.maximum(x[2], 1e-12) for x in rr])  # FD / contrast, per seed
    rel = lambda arr, tru: arr / np.maximum(tru.max(0, keepdims=True), 1e-12)
    S['rate_estimation_n24'] = dict(median_ratio_FD_over_contrast_coarse=float(np.median(ratio[:, :, 0])),
                                    median_ratio_FD_over_contrast_fine=float(np.median(ratio[:, :, 1])),
                                    min_ratio=float(ratio.min()), grid_points=int(len(g)))
    # info-curve RMSE (honest comparison)
    for n in ('24',):
        e = np.array([r['rmse'][n]['entropy'] for r in acc]); c = np.array([r['rmse'][n]['contrast'] for r in acc])
        S['info_curve_rmse_n24'] = dict(entropy_median_max=float(np.median(e.max(1))), contrast_median_max=float(np.median(c.max(1))),
                                        fraction_gamma_contrast_better=float((c < e).mean()))
    # ---- prediction
    rows = []
    for r in allb + nulls:
        rows.append(dict(model=r['model'], N=16, ident=r['seed'], accepted=r['accepted'],
                         measured=r['half_measured']['3.2'], predicted=r['half_predicted']['3.2'], D2=[r['rate0_contrast_norms']['D2_coarse'], r['rate0_contrast_norms']['D2_fine']]))
    L = json.loads((OUT / 'E6A' / 'large_prediction.json').read_text())
    for r in L['rows']:
        rows.append(dict(model=r['model'], N=r['N'], ident=r['ident'], accepted=r['model'] == 'bethe',
                         measured=r['measured_half'], predicted=r['pred_half_grid'], predicted_continuous=r['pred_half_continuous'], predicted_grid=r['pred_half_grid'], D2=r['D2']))
    def logerr(sel, lev):
        v = [abs(np.log(x['predicted'][lev] / x['measured'][lev])) for x in rows if sel(x) and x['measured'][lev] and x['predicted'][lev]]
        return dict(n=len(v), median_abs_log_ratio=float(np.median(v)), max_abs_log_ratio=float(np.max(v)))
    S['prediction'] = dict(
        N16_bethe_all={l: logerr(lambda x: x['N'] == 16 and x['model'] == 'bethe', i) for i, l in enumerate(('coarse', 'fine'))},
        N16_nulls={l: logerr(lambda x: x['N'] == 16 and x['model'] != 'bethe', i) for i, l in enumerate(('coarse', 'fine'))},
        N3264_bethe={l: logerr(lambda x: x['N'] > 16 and x['model'] == 'bethe', i) for i, l in enumerate(('coarse', 'fine'))},
        N3264_two_state={l: logerr(lambda x: x['N'] > 16 and x['model'] == 'two_state', i) for i, l in enumerate(('coarse', 'fine'))},
        N3264_product={l: logerr(lambda x: x['N'] > 16 and x['model'] == 'product', i) for i, l in enumerate(('coarse', 'fine'))},
        grid_steps_N3264=_gridsteps(rows),
        grid_agreement_N3264_bethe=[int(x['predicted_grid'][l] == x['measured'][l]) for x in rows if x['N'] > 16 and x['model'] == 'bethe' for l in (0, 1)],
        rows=rows)
    # rank correlation of fine/coarse time ratio vs contrast-norm ratio across all Bethe instances
    from scipy.stats import spearmanr
    bet = [x for x in rows if x['model'] == 'bethe']
    rat_m = [x['measured'][1] / x['measured'][0] for x in bet]; rat_d = [x['D2'][0] / x['D2'][1] for x in bet]
    S['prediction']['spearman_time_ratio_vs_contrast_ratio_bethe'] = float(spearmanr(rat_m, rat_d)[0])
    # ---- calibration
    cal = [json.loads((OUT / 'E6A' / f'calibration_seed{s}.json').read_text()) for s in N16_ACCEPTED if (OUT / 'E6A' / f'calibration_seed{s}.json').exists()]
    cal_rows = {}
    for c in cal:
        for r in c['rows']:
            key = (r['kind'], r['beta_q'], r['snr_scale'])
            cal_rows.setdefault(key, []).append((max(r['max_abs_identity_gap']), max(r['max_abs_error_vs_exact_information']), max(r['max_abs_z'])))
    S['calibration'] = [dict(kind=k[0], beta_q=k[1], snr_scale=k[2], median_gap=float(np.median([a for a, b, z in v])),
                             max_gap=float(np.max([a for a, b, z in v])), median_true_error=float(np.median([b for a, b, z in v])),
                             median_z=float(np.median([z for a, b, z in v])), z_all=[float(z) for a, b, z in v],
                             err_all=[float(b) for a, b, z in v], n=len(v)) for k, v in cal_rows.items()]
    if cal_rows:
        ex = [z for a, b, z in cal_rows[('exact', 2.5, 1.0)]]
        thr = max(ex)
        mis = [(z, b) for k, v in cal_rows.items() if k[0] != 'exact' for a, b, z in v]
        from scipy.stats import spearmanr
        S['calibration_stats'] = dict(exact_max_z=thr, n_misspecified=len(mis), flagged_above_exact_max=int(sum(z > thr for z, b in mis)),
                                      flagged_true_error_median=float(np.median([b for z, b in mis if z > thr])) if any(z > thr for z, b in mis) else None,
                                      unflagged_true_error_median=float(np.median([b for z, b in mis if z <= thr])),
                                      unflagged_true_error_max=float(np.max([b for z, b in mis if z <= thr])),
                                      spearman_z_vs_true_error=float(spearmanr([z for z, b in mis], [b for z, b in mis])[0]), seeds=[c['seed'] for c in cal])
    json_dump(OUT / 'STAGE6A_SUMMARY.json', S)
    # =========================== figure
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2)); ax = ax.ravel()
    # (a) prediction
    a = ax[0]
    mk = {16: 'o', 32: 's', 64: 'D'}
    for x in rows:
        for lev, col in ((0, CO), (1, FI)):
            if not (x['measured'][lev] and x['predicted'][lev]): continue
            if x['model'] == 'bethe':
                face = col if x['accepted'] else 'white'
                a.scatter(x['measured'][lev], x['predicted'][lev], marker=mk[x['N']], s=34, facecolor=face, edgecolor=col, lw=1.1, zorder=3)
            else:
                a.scatter(x['measured'][lev], x['predicted'][lev], marker='^' if x['model'] == 'product' else 'v', s=34,
                          facecolor='white', edgecolor=col, lw=.9, zorder=2, alpha=.8)
    lim = [0.015, 3]
    a.plot(lim, lim, ':', color='k', lw=.8); a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel(r'measured half-information $\gamma^{1/2}$ (N>16: on 21-pt grid)'); a.set_ylabel(r'predicted $\gamma^{1/2}$ (centroid channel)')
    a.set_title('(a) Clocks from equilibrium geometry', fontsize=10, loc='left')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', mfc=CO, mec=CO, label='coarse'), Line2D([], [], marker='o', ls='', mfc=FI, mec=FI, label='fine | coarse'),
         Line2D([], [], marker='o', ls='', mfc='k', mec='k', ms=5, label='Bethe N=16 (open: failed gate)'),
         Line2D([], [], marker='s', ls='', mfc='w', mec='k', ms=5, label='Bethe N=32'), Line2D([], [], marker='D', ls='', mfc='w', mec='k', ms=5, label='Bethe N=64'),
         Line2D([], [], marker='v', ls='', mfc='w', mec='k', ms=5, label='two-state null (N=16,32,64)'), Line2D([], [], marker='^', ls='', mfc='w', mec='k', ms=5, label='product null (N=16,32,64)')]
    a.legend(handles=h, fontsize=6.5, loc='upper left', frameon=False)
    # (b) budget
    b = ax[1]
    for k, col, lab in (('coarse', CO, 'coarse'), ('fine', FI, 'fine | coarse'), ('within', GR, 'within leaf')):
        med = np.median(frac[k], 0); lo, hi = np.percentile(frac[k], [25, 75], 0)
        b.plot(g[1:], med[1:], color=col, lw=2, label=lab); b.fill_between(g[1:], lo[1:], hi[1:], color=col, alpha=.15, lw=0)
    b.set_xscale('log'); b.set_ylim(0, 1); b.set_xlabel(r'$\gamma$'); b.set_ylabel('fraction of total information rate')
    b.set_title('(b) Information budget (exact N=16)', fontsize=10, loc='left'); b.legend(fontsize=7, frameon=False, loc='center left')
    # (c) rate estimation
    c = ax[2]
    for lev, col in ((0, CO), (1, FI)):
        fd = np.median([x[1][:, lev] / x[3][:, lev].max() for x in rr], 0)
        cc = np.median([x[2][:, lev] / x[3][:, lev].max() for x in rr], 0)
        c.plot(gm, fd, '--', color=col, lw=1.4); c.plot(gm, cc, '-', color=col, lw=2)
    c.set_xscale('log'); c.set_yscale('log'); c.set_xlabel(r'$\gamma$'); c.set_ylabel('rate RMSE / peak rate (24 paths)')
    c.set_title('(c) Rate profile: contrast vs finite difference', fontsize=10, loc='left')
    c.legend(handles=[Line2D([], [], color='k', ls='--', label='finite difference of entropy curve'), Line2D([], [], color='k', lw=2, label='contrast (compensator) form'),
                      Line2D([], [], color=CO, lw=2, label='coarse'), Line2D([], [], color=FI, lw=2, label='fine | coarse')], fontsize=7, frameon=False)
    # (d) calibration
    d = ax[3]
    if S['calibration']:
        ex = [r for r in S['calibration'] if r['kind'] == 'exact'][0]
        d.axhspan(0, max(ex['z_all']), color=GR, alpha=.15, lw=0)
        d.text(.012, max(ex['z_all']) * .5, 'exact filter (all seeds)', fontsize=7, color='k', va='center', transform=d.get_yaxis_transform())
        for r in S['calibration']:
            if r['kind'] == 'exact': continue
            x = r['beta_q'] / 2.5 if r['kind'] == 'beta' else r['snr_scale']
            col = 'k' if r['kind'] == 'beta' else FI
            d.scatter([x] * len(r['z_all']), r['z_all'], s=14 + 400 * np.array(r['err_all']), facecolor='none', edgecolor=col, lw=.9)
        d.set_xscale('log'); d.set_xticks([.7, .8, .9, 1.1, 1.25]); d.set_xticklabels(['0.7', '0.8', '0.9', '1.1', '1.25 / 1.3']); d.minorticks_off()
        d.set_xlabel(r"mis-specification ($\beta'/\beta_J$: black; field scale $\lambda$: orange)")
        d.set_ylabel(r'identity gap $\max_\gamma|\Delta_q|$ / path s.e.')
        d.set_title('(d) Reference-free calibration test', fontsize=10, loc='left')
        d.legend(handles=[Line2D([], [], marker='o', ls='', mfc='none', mec='k', label='symbol size $\\propto$ true info. error')], fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout()
    for ext in ('pdf', 'png'): fig.savefig(FIG / f'E6A_contrast_algorithms.{ext}', dpi=200)
    print(json.dumps({k: v for k, v in S.items() if k != 'prediction'}, indent=1, default=str)[:3000])
    print({k: v for k, v in S['prediction'].items() if k != 'rows'})


if __name__ == '__main__':
    main()
