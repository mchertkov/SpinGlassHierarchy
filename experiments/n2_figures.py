"""Figures and summaries for campaign N2: per-path decision counting (Fig3_paths) and protocol/discretization (Fig7_protocol).
usage: python experiments/n2_figures.py
"""
from pathlib import Path
import sys, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / 'N2'; FIG = ROOT / 'paper' / 'figures'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
LC = ['#1f4e79', '#c55a11', '#2e8b57', '#7f3f98']; LN = ['sector', 'glassy level 1', 'glassy level 2', 'glassy level 3']


def recount(z, thr=0.9, f=2.0):
    """commitments per path and distinct events from stored posteriors (threshold thr, merge factor f)"""
    g, pp, sp = z['gamma'], z['pp'], z['splits']; n, _, L = pp.shape
    nc = np.zeros(n, int); ne = np.zeros(n, int); ratios = []
    for i in range(n):
        c = []
        for l in range(L):
            if not sp[i, l]: continue
            h = np.flatnonzero(pp[i, :, l] >= thr)
            if len(h): c.append((l, g[h[0]]))
        nc[i] = len(c); cs = sorted(x[1] for x in c)
        ne[i] = 0 if not cs else 1 + sum(b / a >= f for a, b in zip(cs[:-1], cs[1:]))
        for (l1, a), (l2, b) in zip(c[:-1], c[1:]):
            ratios.append((l2, b / a))
    return nc, ne, ratios


def load(family, N, beta, glassy_only=True):
    out = []
    for f in sorted(glob.glob(str(RES / f'paths_{family}_N{N}_b{beta}_s*.json'))):
        o = json.load(open(f))
        if glassy_only and o['n_levels_glassy'] < 1: continue
        z = np.load(f.replace('.json', '.npz')); out.append((o, z))
    return out


def anatomy24():
    z = np.load(ROOT / 'results' / 'N1' / 'bethe_N24_b2.5_s2_anatomy.npz'); g = z['gamma']; w = z['w'] / z['w'].sum(); n = len(w)
    nc, ev = [], []
    for i in range(n):
        c = []
        for l in range(4):
            P = z[f'post{l}']; pl = z[f'plant{l}']
            if l > 0 and abs(P[i, 0, pl[i]] - z[f'post{l-1}'][i, 0, z[f'plant{l-1}'][i]]) < 1e-9: continue
            h = np.flatnonzero(P[i, :, pl[i]] >= 0.9)
            if len(h): c.append(g[h[0]])
        cs = sorted(c); nc.append(len(c)); ev.append(1 + sum(b / a >= 2 for a, b in zip(cs[:-1], cs[1:])))
    return float(w @ np.array(nc)), float(w @ np.array(ev))


def paths_fig():
    fig = plt.figure(figsize=(13, 3.6)); gs = fig.add_gridspec(1, 4, wspace=.33)
    # (a) one trajectory
    z = np.load(RES / 'paths_bethe_N16_b2.5_s20.npz'); g = z['gamma']; i = 0
    ax = fig.add_subplot(gs[0]); ax.set_title('(a) one trajectory ($N=16$, $\\beta_J=2.5$)')
    for l in range(z['pp'].shape[2]):
        if not z['splits'][i, l]: continue
        ax.plot(16 * g[1:], z['pp'][i, 1:, l], color=LC[l], lw=1.6, label=LN[l])
        h = np.flatnonzero(z['pp'][i, :, l] >= 0.9)
        if len(h): ax.plot(16 * g[h[0]], 0.9, 'v', color=LC[l], ms=6)
    ax.axhline(0.9, ls=':', c='k', lw=.6); ax.set_xscale('log'); ax.set_xlabel('$\\tau=N\\gamma$'); ax.set_ylabel('posterior mass of the planted class')
    ax.legend(frameon=False, fontsize=6.8, loc='lower right'); ax.set_ylim(0, 1.03)
    ax.text(0.03, 0.95, 'triangles: commitments', transform=ax.transAxes, fontsize=6.8, color='0.35', va='top')
    # (b) histogram commitments vs distinct events (all glasses N=16, beta_J >= 1)
    ax = fig.add_subplot(gs[1]); ax.set_title('(b) per trajectory: commitments\nvs distinct decision events ($N=16$)')
    NC, NE, W = [], [], []
    for b in (1.0, 1.5, 2.0, 2.5):
        for o, z in load('bethe', 16, b):
            nc, ne, _ = recount(z); w = z['w'] / len(load_cache_n(o)) if False else z['w']
            NC.append(nc); NE.append(ne); W.append(z['w'])
    NC, NE, W = np.concatenate(NC), np.concatenate(NE), np.concatenate(W); W = W / W.sum()
    bins = np.arange(0.5, 5.5)
    ax.hist(NC, bins=bins, weights=W, color='0.7', label='commitments (levels on the branch)', rwidth=.85)
    ax.hist(NE, bins=bins, weights=W, histtype='step', color=LC[1], lw=2, label='distinct events ($f=2$)')
    ax.set_xlabel('number per trajectory'); ax.set_ylabel('fraction of trajectories'); ax.legend(frameon=False, fontsize=6.8)
    ax.set_title('(b) per trajectory: commitments vs distinct\nevents ($N=16$, instances with glassy levels)')
    summ = dict(N16_glass=dict(mean_commit=float(W @ NC), mean_events=float(W @ NE), frac_events1=float(W[NE == 1].sum()),
                               frac_events2=float(W[NE == 2].sum()), frac_events3p=float(W[NE >= 3].sum())))
    # (c) means vs beta_J
    ax = fig.add_subplot(gs[2]); ax.set_title('(c) mean per trajectory vs coupling')
    for fam, N, mk, col in [('bethe', 16, 'o', '#1f4e79'), ('bethe', 20, 's', '#2e75b6')]:
        xb, mc, me = [], [], []
        for b in (1.0, 1.5, 2.0, 2.5):
            D = load(fam, N, b)
            if not D: continue
            c_, e_, w_ = [], [], []
            for o, z in D:
                nc, ne, _ = recount(z); c_.append(nc); e_.append(ne); w_.append(z['w'] / len(D))
            c_, e_, w_ = map(np.concatenate, (c_, e_, w_)); w_ = w_ / w_.sum()
            xb.append(b); mc.append(float(w_ @ c_)); me.append(float(w_ @ e_))
        lab = f'{"Bethe" if fam == "bethe" else "ferro (|J|)"} $N={N}$'
        ax.plot(xb, mc, '-' + mk, color=col, ms=4, label=lab + ': commitments'); ax.plot(xb, me, ':' + mk, color=col, ms=4, mfc='none', label=lab + ': events')
        summ[f'{fam}_N{N}_vs_beta'] = dict(beta=xb, mean_commit=mc, mean_events=me)
    c24, e24 = anatomy24(); ax.plot([2.5], [c24], '*', color='#9dc3e6', ms=10, label='$N=24$ instance of Fig. 2: commitments')
    ax.plot([2.5], [e24], '*', color='#9dc3e6', ms=10, mfc='none', label='$N=24$ instance: events'); summ['N24_anatomy'] = dict(commit=c24, events=e24)
    ax.set_xlabel('$\\beta_J$'); ax.set_ylim(0.8, 4.4); ax.legend(frameon=False, fontsize=5.8, loc='upper left')
    ax.set_title('(c) mean per trajectory\n(instances with glassy levels)')
    # (d) ratios of successive commitment times
    ax = fig.add_subplot(gs[3]); ax.set_title('(d) spacing of successive commitments\n(ratio of $\\gamma$; $N=16$, $\\beta_J\\geq1$)')
    R = {1: [], 2: []}
    for b in (1.0, 1.5, 2.0, 2.5):
        for o, z in load('bethe', 16, b):
            for l2, r in recount(z)[2]:
                R[min(l2, 2)].append(r)
    bins = np.geomspace(1, 30, 16)
    ax.hist(R[1], bins=bins, density=True, alpha=.6, color=LC[1], label='sector $\\to$ glassy 1')
    ax.hist(R[2], bins=bins, density=True, histtype='step', lw=2, color=LC[2], label='glassy $\\to$ deeper glassy')
    ax.axvline(25, color='0.5', ls='--', lw=.8); ax.text(23, ax.get_ylim()[1] * 0.9, 'one window\n(factor 25)', ha='right', fontsize=6.6, color='0.4')
    ax.set_xscale('log'); ax.set_xlabel('ratio of commitment times'); ax.legend(frameon=False, fontsize=6.8, loc='center right')
    summ['ratios'] = {k: dict(n=len(v), median=float(np.median(v)), frac_lt_1p5=float(np.mean(np.array(v) < 1.5)),
                              frac_ge_2=float(np.mean(np.array(v) >= 2))) for k, v in R.items()}
    # robustness of the mean number of distinct events
    rob = {}
    for thr in (0.8, 0.9, 0.95):
        for f in (1.5, 2.0, 3.0):
            E_, W_ = [], []
            for b in (1.0, 1.5, 2.0, 2.5):
                for o, z in load('bethe', 16, b):
                    E_.append(recount(z, thr, f)[1]); W_.append(z['w'])
            E_, W_ = np.concatenate(E_), np.concatenate(W_); W_ = W_ / W_.sum(); rob[f'thr{thr}_f{f}'] = float(W_ @ E_)
    summ['robustness_mean_events'] = rob
    # recrossings
    rc = {0: [0.0, 0.0], 1: [0.0, 0.0]}
    for b in (1.0, 1.5, 2.0, 2.5):
        for o, z in load('bethe', 16, b):
            cm, rr, w = np.array(o['commit']), np.array(o['recross']), z['w']
            for l in (0, 1):
                if cm.shape[1] > l:
                    m = cm[:, l] > 0; rc[l][0] += float(w[m & rr[:, l]].sum()); rc[l][1] += float(w[m].sum())
    summ['recross_frac_sector_glassy1'] = [rc[l][0] / max(rc[l][1], 1e-12) for l in (0, 1)]
    fig.savefig(FIG / 'Fig3_paths.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Fig3_paths.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    json.dump(summ, open(RES / 'paths_summary.json', 'w'), indent=1)
    return summ


def load_cache_n(o):
    return [0]


def gam(S, t):
    t = np.asarray(t, float)
    return t / (1 - t) if S == 0 else S * np.sinh(S * t) / (np.sinh(S) * np.sinh(S * (1 - t)))


def protocol_fig():
    D = [json.load(open(f)) for f in sorted(glob.glob(str(RES / 'disc_N16_*.json')))]
    protos = [('follmer', '$\\nu=0$', '-', 'k'), ('tuned', 'tuned $\\nu$ (decisions at $t=1/2$)', '--', '#c55a11'), ('stiff', '$\\nu=144$', ':', '#2e75b6')]
    Ks = sorted(int(k) for k in D[0]['res']['follmer'])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4)); summ = dict(n_instances=len(D))
    for name, lab, ls, col in protos:
        cls = np.array([[d['res'][name][str(K)]['tv'][min(1, len(d['res'][name][str(K)]['tv']) - 1)] - d['floor'][min(1, len(d['floor']) - 1)] for K in Ks] for d in D])
        fin = np.array([[d['res'][name][str(K)]['tv'][-1] - d['floor'][-1] for K in Ks] for d in D])
        en = np.array([[d['res'][name][str(K)]['logmu_bias'] for K in Ks] for d in D])
        axs[0].plot(Ks, np.median(fin, 0), ls=ls, marker='o', ms=3.5, color=col, label=lab)
        axs[1].plot(Ks, -np.median(en, 0), ls=ls, marker='o', ms=3.5, color=col, label=lab)
        S = D[0]['S'][name] if name != 'tuned' else np.median([d['S']['tuned'] for d in D])
        loc = [int((gam(S, np.arange(K) / K) >= 1).sum()) for K in Ks]
        axs[2].plot(Ks, np.array(loc) / np.array(Ks), ls=ls, marker='o', ms=3.5, color=col, label=lab)
        summ[name] = dict(K=Ks, class_tv_excess_median=np.median(fin, 0).tolist(), glassy1_tv_excess_median=np.median(cls, 0).tolist(),
                          energy_bias_median=np.median(en, 0).tolist(), local_fraction=(np.array(loc) / np.array(Ks)).tolist(),
                          S_median=float(S))
    axs[0].axhline(0, color='0.6', lw=.7); axs[0].set_xscale('log', base=2); axs[0].set_xlabel('number of steps $K$')
    axs[0].set_title('(a) class error of the endpoints\n(total variation above sampling noise)')
    axs[1].set_xscale('log', base=2); axs[1].set_yscale('log'); axs[1].set_xlabel('number of steps $K$')
    axs[1].set_title('(b) energy error of the endpoints\n($-$bias of $\\log\\mu$, in units of its std)')
    axs[2].set_xscale('log', base=2); axs[2].set_xlabel('number of steps $K$'); axs[2].set_ylim(0, 0.6)
    axs[2].set_title('(c) fraction of steps in the local stage\n($\\gamma\\geq1$)')
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'Fig08_protocol.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Fig08_protocol.png', dpi=150, bbox_inches='tight')
    plt.close(fig); json.dump(summ, open(RES / 'protocol_summary.json', 'w'), indent=1)
    return summ




def partial_fig():
    """Fig. 3 (v5): partial commitments of one trajectory, their ensemble median, and the counting statistics."""
    zp = np.load(ROOT / 'results' / 'N3' / 'partial_N24.npz'); g = zp['gamma']; N = 24; tau = N * g; maps = zp['maps']; L = maps.shape[0]
    PF = zp['post_fine']; n = PF.shape[0]; K = PF.shape[2]
    # conditional partial commitments of the planted branch
    P = [zp[f'post{l}'] for l in range(L)]; pl = [zp[f'plant{l}'] for l in range(L)]
    pp = np.stack([P[l][np.arange(n), :, pl[l]] for l in range(L)], -1)            # (n, g, L) planted-class mass
    q = pp.copy(); q[:, :, 1:] = pp[:, :, 1:] / np.maximum(pp[:, :, :-1], 1e-12)
    split = np.ones((n, L), bool)
    for l in range(1, L): split[:, l] = np.abs(P[l][:, 0, :][np.arange(n), pl[l]] - P[l - 1][:, 0, :][np.arange(n), pl[l - 1]]) > 1e-9
    # pick a trajectory whose branch splits at every level and that commits in >= 2 distinct events
    def events(i):
        c = sorted(g[np.argmax(pp[i, :, l] >= 0.9)] for l in range(L) if split[i, l] and (pp[i, :, l] >= 0.9).any())
        return 1 + sum(b / a >= 2 for a, b in zip(c[:-1], c[1:]))
    cand = [i for i in range(n) if split[i].all() and events(i) >= 3] or [i for i in range(n) if split[i].all()]
    i0 = cand[0]
    fig = plt.figure(figsize=(13, 7.0)); gs = fig.add_gridspec(2, 3, hspace=.48, wspace=.3)
    # (a) stacked distribution over finest classes, coloured by tree
    ax = fig.add_subplot(gs[0, 0]); ax.set_title('(a) one trajectory: probability of every class')
    sec = maps[0]; l1 = maps[1]
    order = np.lexsort((maps[-1], maps[2], l1, -sec))
    import matplotlib.colors as mc
    fams = {}
    cols = []
    for c in order:
        key = (sec[c], l1[c]); fams.setdefault(key, 0); fams[key] += 1
        base = {True: ['#1f4e79', '#2e8b57'], False: ['#b0b0b0', '#d9d9d9']}[bool(sec[c] == sec.max())][int(l1[c]) % 2]
        rgb = np.array(mc.to_rgb(base)); cols.append(np.clip(rgb * (0.75 + 0.18 * fams[key]), 0, 1))
    ax.stackplot(tau[1:], PF[i0, 1:, order], colors=cols, edgecolor='white', lw=0.3)
    fc = int(np.argmax(PF[i0, -1])); ax.text(tau[-1] * 0.9, 0.5, "endpoint's\nclass", ha='right', fontsize=6.5, color='white')
    ax.set_xscale('log'); ax.set_xlim(tau[1], tau[-1]); ax.set_ylim(0, 1); ax.set_xlabel('$\\tau=N\\gamma$'); ax.set_ylabel('posterior class probabilities')
    # (b) conditional partial commitments of this trajectory
    ax = fig.add_subplot(gs[0, 1]); ax.set_title('(b) same trajectory: conditional\npartial commitments $q_\\ell$')
    for l in range(L):
        ax.plot(tau[1:], q[i0, 1:, l], color=LC[l], lw=1.8, label=LN[l])
        h = np.flatnonzero(pp[i0, :, l] >= 0.9)
        if len(h): ax.plot(tau[h[0]], q[i0, h[0], l], 'v', color=LC[l], ms=6)
    ax.axhline(0.9, color='k', ls=':', lw=.6); ax.set_xscale('log'); ax.set_ylim(0, 1.03); ax.set_xlabel('$\\tau=N\\gamma$')
    ax.legend(frameon=False, fontsize=6.6, loc='lower right'); ax.text(0.03, 0.04, 'triangles: commitments', transform=ax.transAxes, fontsize=6.5, color='0.35')
    # (c) ensemble median of the conditional partial commitments (branches that split)
    ax = fig.add_subplot(gs[0, 2]); ax.set_title('(c) all trajectories: median $q_\\ell$\n(interquartile band)')
    for l in range(L):
        m = split[:, l]; Q = q[m][:, 1:, l]
        ax.plot(tau[1:], np.median(Q, 0), color=LC[l], lw=1.8, label=LN[l]); ax.fill_between(tau[1:], np.quantile(Q, .25, 0), np.quantile(Q, .75, 0), color=LC[l], alpha=.15)
    ax.set_xscale('log'); ax.set_ylim(0, 1.03); ax.set_xlabel('$\\tau=N\\gamma$'); ax.legend(frameon=False, fontsize=6.6, loc='lower right')
    # (d) commitments vs events, N=16
    ax = fig.add_subplot(gs[1, 0]); ax.set_title('(d) per trajectory: commitments and\ndistinct events ($N=16$)')
    NC, NE, W = [], [], []
    for b in (1.0, 1.5, 2.0, 2.5):
        for o, z in load('bethe', 16, b):
            nc, ne, _ = recount(z); NC.append(nc); NE.append(ne); W.append(z['w'])
    NC, NE, W = np.concatenate(NC), np.concatenate(NE), np.concatenate(W); W = W / W.sum()
    bins = np.arange(0.5, 5.5)
    ax.hist(NC, bins=bins, weights=W, color='0.75', label='commitments', rwidth=.85)
    ax.hist(NE, bins=bins, weights=W, histtype='step', color=LC[1], lw=2, label='distinct events ($f=2$)')
    ax.set_xlabel('number per trajectory'); ax.set_ylabel('fraction of trajectories'); ax.legend(frameon=False, fontsize=6.8)
    # (e) means vs beta, N=16, 20, N=24 instance
    ax = fig.add_subplot(gs[1, 1]); ax.set_title('(e) mean per trajectory vs coupling')
    for fam, NN, mk, col in [('bethe', 16, 'o', '#1f4e79'), ('bethe', 20, 's', '#2e75b6')]:
        xb, mcm, me = [], [], []
        for b in (1.0, 1.5, 2.0, 2.5):
            D = load(fam, NN, b)
            if not D: continue
            c_, e_, w_ = [], [], []
            for o, z in D:
                nc, ne, _ = recount(z); c_.append(nc); e_.append(ne); w_.append(z['w'] / len(D))
            c_, e_, w_ = map(np.concatenate, (c_, e_, w_)); w_ = w_ / w_.sum(); xb.append(b); mcm.append(w_ @ c_); me.append(w_ @ e_)
        ax.plot(xb, mcm, '-' + mk, color=col, ms=4, label=f'$N={NN}$ commitments'); ax.plot(xb, me, ':' + mk, color=col, ms=4, mfc='none', label=f'$N={NN}$ events')
    c24, e24 = anatomy24(); ax.plot([2.5], [c24], '*', color='#9dc3e6', ms=10, label='$N=24$ instance'); ax.plot([2.5], [e24], '*', color='#9dc3e6', ms=10, mfc='none')
    ax.set_xlabel('$\\beta_J$'); ax.set_ylim(0.8, 4.2); ax.legend(frameon=False, fontsize=6.0, loc='upper left', ncol=2)
    # (f) spacing of successive commitments
    ax = fig.add_subplot(gs[1, 2]); ax.set_title('(f) spacing of successive commitments\n($N=16$, $\\beta_J\\geq1$)')
    R = {1: [], 2: []}
    for b in (1.0, 1.5, 2.0, 2.5):
        for o, z in load('bethe', 16, b):
            for l2, r in recount(z)[2]: R[min(l2, 2)].append(r)
    bins = np.geomspace(1, 30, 16)
    ax.hist(R[1], bins=bins, density=True, alpha=.6, color=LC[1], label='sector $\\to$ glassy 1')
    ax.hist(R[2], bins=bins, density=True, histtype='step', lw=2, color=LC[2], label='glassy $\\to$ deeper glassy')
    ax.axvline(25, color='0.5', ls='--', lw=.8); ax.text(23, ax.get_ylim()[1] * 0.9, 'one window\n(factor 25)', ha='right', fontsize=6.4, color='0.4')
    ax.set_xscale('log'); ax.set_xlabel('ratio of commitment times'); ax.legend(frameon=False, fontsize=6.6, loc='center right')
    fig.savefig(FIG / 'Fig03_partial.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Fig03_partial.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    med = {l: float(np.interp(0.5, np.median(q[split[:, l]][:, 1:, l], 0), tau[1:])) for l in range(L)}
    json.dump(dict(trajectory=int(i0), median_q_half_tau=med), open(RES / 'partial_summary.json', 'w'), indent=1)
    return i0, med


if __name__ == '__main__':
    # Fig. 3 (partial commitments) and Fig. 8 (protocol); 'paths' draws an earlier per-trajectory figure not used in the paper
    what = sys.argv[1:] or ['partial', 'protocol']
    if 'partial' in what: partial_fig()
    if 'protocol' in what: print(json.dumps(protocol_fig(), indent=0)[:2500])
    if 'paths' in what: print(json.dumps(paths_fig(), indent=0)[:2500])
