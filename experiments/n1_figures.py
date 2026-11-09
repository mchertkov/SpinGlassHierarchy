"""Figures for campaign N1 (exact level-resolved decisions): anatomy, scan, binary-channel law.
usage: python experiments/n1_figures.py [anatomy temp size law]  -> paper/figures/Fig02_anatomy, Fig04_temperature, Fig05_sizes, Fig06_law
"""
from pathlib import Path
import sys, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import exacttree as E
RES = ROOT / 'results' / 'N1'; FIG = ROOT / 'paper' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
LC = ['#1f4e79', '#c55a11', '#2e8b57', '#7f3f98']           # sector, glassy 1, 2, 3
LN = ['sector ($\\mathbb{Z}_2$)', 'glassy level 1', 'glassy level 2', 'glassy level 3']

xs = np.linspace(-8, 8, 4001); wx = np.exp(-xs ** 2 / 2); wx /= wx.sum()
def Ibin(s):
    s = np.atleast_1d(s)[:, None]; u = s + np.sqrt(s) * xs[None, :]
    return (np.log(2) - (np.logaddexp(0, -2 * u) * wx).sum(1)) / np.log(2)


def pred_curve(g, chans):
    w = np.array([c['pi'] * c['H'] for c in chans]); w = w / w.sum()
    return sum(wi * Ibin(g * c['kappa']) for wi, c in zip(w, chans))


def tree_nodes(info, nlev):
    """Rebuild node masses per level from split records (quotient masses)."""
    levels = [[(1.0, None)]]
    for l in range(nlev):
        sp = {s['parent']: s for s in info[l]} if l < len(info) else {}
        nxt = []
        for c, (m, _) in enumerate(levels[-1]):
            if c in sp: nxt += [(sp[c]['masses'][0], c), (sp[c]['masses'][1], c)]
            else: nxt.append((m, c))
        levels.append(nxt)
    return levels


def anatomy(name='bethe_N24_b2.5_s2_anatomy', out='Fig02_anatomy', title=None):
    o = json.load(open(RES / f'{name}.json')); z = np.load(RES / f'{name}.npz')
    N = o['N']; g = np.array(o['gamma']); tau = N * g; L = len(o['n_classes'])
    w = z['w'] if 'w' in z.files else np.full(len(z['Y']), 1 / len(z['Y']))
    fig = plt.figure(figsize=(11, 6.6)); gs = fig.add_gridspec(2, 3, hspace=.45, wspace=.3)
    # (a) tree with recursive layout (x of leaves evenly spaced, parents at the mean of their children)
    ax = fig.add_subplot(gs[0, 0]); ax.axis('off'); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.12, 1.08)
    ax.set_title(title or f'(a) nested classes inside one sector ($N={N}$)')
    lv = tree_nodes(o['level_info'], L - 1); X = [None] * len(lv)
    n_last = len(lv[-1]); X[-1] = list(np.linspace(0.06, 0.94, n_last)) if n_last > 1 else [0.5]
    for l in range(len(lv) - 2, -1, -1):
        X[l] = [float(np.mean([X[l + 1][j] for j, (_, p) in enumerate(lv[l + 1]) if p == c])) for c in range(len(lv[l]))]
    dy = 0.95 / max(1, len(lv) - 1)
    for l, nodes in enumerate(lv):
        y = 1.0 - dy * l
        for j, (m, par) in enumerate(nodes):
            carried = par is not None and sum(1 for _, p in nodes if p == par) == 1
            if par is not None: ax.plot([X[l - 1][par], X[l][j]], [y + dy, y], color='0.65', lw=0.8, zorder=0, ls=':' if carried else '-')
            col = LC[min(l, 3)] if l > 0 else '0.35'
            if carried: ax.scatter(X[l][j], y, s=700 * m + 15, facecolor='white', edgecolor=col, lw=1.2, zorder=2)
            else: ax.scatter(X[l][j], y, s=700 * m + 15, color=col, zorder=2, edgecolor='white', lw=.5)
            if l > 0: ax.text(X[l][j], y - 0.085, f'{m:.2f}', ha='center', fontsize=6.3)
    for l, sp in enumerate(o['level_info']):
        for s_ in sp:
            ax.text(X[l][s_['parent']] + 0.055, 1.0 - dy * l + 0.01, f"$\\Delta$={s_['kappa']:.1f}", fontsize=6.3, color=LC[l + 1])
    # (b) one path
    ax = fig.add_subplot(gs[0, 1]); ax.set_title("(b) commitment to the endpoint's class")
    i0 = int(np.argmax([((z['post1'][i, :, z['plant1'][i]] > .9).argmax() > 0) and (z['post1'][i, :, z['plant1'][i]] < .8)[(z['post1'][i, :, z['plant1'][i]] > .9).argmax():].any() for i in range(len(z['Y']))]))
    for l in range(L):
        P = z[f'post{l}']; pl = z[f'plant{l}']; pp = P[np.arange(len(pl)), :, pl]
        for i in range(12): ax.plot(tau[1:], pp[i, 1:], color=LC[l], lw=0.4, alpha=0.35)
        ax.plot(tau[1:], pp[i0, 1:], color=LC[l], lw=1.8, label=LN[l])
    ax.set_xscale('log'); ax.set_xlabel('$\\tau=N\\gamma$'); ax.set_ylim(0, 1.02); ax.axhline(.9, ls=':', c='k', lw=.6)
    ax.legend(loc='upper left', frameon=True, framealpha=.9, fontsize=6.5)
    # (c) information per level with predictions
    ax = fig.add_subplot(gs[0, 2]); ax.set_title('(c) conditional information per level')
    gg = np.geomspace(g[1], g[-1], 200)
    for l in range(L):
        f = np.array(o['frac'][l]); ax.plot(tau, f, 'o', ms=3.5, color=LC[l], label=LN[l])
        ax.plot(N * gg, pred_curve(gg, o['channels'][l]), '--', color=LC[l], lw=1)
        ci = o.get('half_gamma_ci', [None] * L)[l]
        if ci: ax.plot([N * ci[0], N * ci[1]], [0.5, 0.5], color=LC[l], lw=3, alpha=.5)
    ax.set_xscale('log'); ax.set_xlabel('$\\tau=N\\gamma$'); ax.set_ylabel('fraction of $\\mathrm{H}(C_\\ell|C_{\\ell-1})$')
    ax.text(0.03, 0.93, 'symbols: exact\ndashed: binary channels\n$\\sum_P w_P I_{\\rm bin}(\\gamma\\lambda_P)$', transform=ax.transAxes, fontsize=7, va='top')
    # (d) top-direction alignment
    ax = fig.add_subplot(gs[1, 0]); ax.set_title("(d) leading posterior direction vs contrasts\non the endpoint's branch")
    c = np.array(o['top_cos_median'])
    for l in range(L): ax.plot(tau[1:], c[1:, l], '-o', ms=2.5, color=LC[l], label=LN[l])
    ax.set_xscale('log'); ax.set_xlabel('$\\tau=N\\gamma$'); ax.set_ylabel('median $|\\cos(v_{\\max}(C_\\gamma),d_\\ell)|$'); ax.set_ylim(0, 1.02)
    # (e) top eigenvalues
    ax = fig.add_subplot(gs[1, 1]); ax.set_title('(e) leading eigenvalues of $C_\\gamma$ (median)')
    lam = np.array(o['top_lam_median'])
    for k in range(3): ax.plot(tau[1:], lam[1:, k], '-', color=['k', '0.4', '0.7'][k], label=f'$\\lambda_{k+1}(C_\\gamma)$')
    for l in range(L):
        for ch in o['channels'][l]: ax.axhline(ch['kappa'], color=LC[l], lw=.6, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('$\\tau=N\\gamma$'); ax.legend(frameon=False)
    ax.text(0.03, 0.05, 'dotted: $\\lambda_P$ of the parents', transform=ax.transAxes, fontsize=7, color='0.3')
    # (f) self-overlap rate
    ax = fig.add_subplot(gs[1, 2]); ax.set_title('(f) localization: $\\bar q$ and $d\\bar q/d\\log\\tau$')
    q = np.array(o['qbar']); ax.plot(tau[1:], q[1:], 'k-', label='$\\bar q$')
    lt = np.log(tau[1:]); d = np.gradient(q[1:], lt); ax.plot(tau[1:], d / d.max(), color='0.5', label='rate (normalized)')
    for l in range(L):
        th = o['half_gamma'][l]
        if th: ax.axvline(N * th, color=LC[l], lw=1, ls='--')
    ax.set_xscale('log'); ax.set_xlabel('$\\tau=N\\gamma$'); ax.legend(frameon=False, loc='upper left')
    fig.savefig(FIG / f'{out}.pdf', bbox_inches='tight'); fig.savefig(FIG / f'{out}.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    return o


def load_scan(family, N):
    out = {}
    for f in glob.glob(str(RES / f'{family}_N{N}_b*_s*.json')):
        if 'anatomy' in f: continue
        o = json.load(open(f)); b = float(f.split('_b')[1].split('_s')[0]); out.setdefault(b, []).append(o)
    return dict(sorted(out.items()))


def first_kappa(o):
    c = [x for x in o.get('candidates', []) if x['level'] == 1]
    return c[0]['kappa'] if c else np.nan


def scan_fig():
    fams = [('bethe', 16, 'o', '#1f4e79'), ('bethe', 20, 's', '#2e75b6'), ('bethe', 24, 'D', '#9dc3e6'), ('sk', 16, '^', '#c55a11'),
            ('ferro', 16, 'x', '0.45'), ('ferro', 20, '+', '0.65')]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2)); summary = {}
    for fam, N, mk, col in fams:
        S = load_scan(fam, N)
        if not S: continue
        bs = np.array(list(S)); lab = f'{dict(bethe="Bethe", sk="SK", ferro="ferro (|J|)")[fam]} $N={N}$'
        nl = [np.mean([o['n_levels_glassy'] for o in S[b]]) for b in bs]
        f1 = [np.mean([o['n_levels_glassy'] >= 1 for o in S[b]]) for b in bs]
        f2 = [np.mean([o['n_levels_glassy'] >= 2 for o in S[b]]) for b in bs]
        k1 = [np.nanmedian([first_kappa(o) / N for o in S[b]]) for b in bs]
        l1 = [np.median([o['lam'][0] / N for o in S[b]]) for b in bs]
        r10 = [np.nanmedian([o['half_gamma'][1] / o['half_gamma'][0] for o in S[b] if o['n_levels_glassy'] >= 1 and o['half_gamma'][1]] or [np.nan]) for b in bs]
        r21 = [np.nanmedian([o['half_gamma'][2] / o['half_gamma'][1] for o in S[b] if o['n_levels_glassy'] >= 2 and o['half_gamma'][2]] or [np.nan]) for b in bs]
        axs[0].plot(bs, f1, '-' + mk, color=col, label=lab, ms=4); axs[0].plot(bs, f2, ':' + mk, color=col, ms=3, mfc='none')
        axs[1].plot(bs, k1, '-' + mk, color=col, ms=4); axs[2].plot(bs, l1, '-' + mk, color=col, ms=4)
        axs[3].plot(bs, r10, '-' + mk, color=col, ms=4); axs[3].plot(bs, r21, ':' + mk, color=col, ms=3, mfc='none')
        summary[f'{fam}_N{N}'] = dict(betas=bs.tolist(), n=[len(S[b]) for b in bs], mean_levels=nl, frac_ge1=f1, frac_ge2=f2,
                                     kappa1_over_N=k1, lam1_over_N=l1, ratio_l1_sector=r10, ratio_l2_l1=r21)
    axs[0].set_title('(a) instances with $\\geq1$ (solid), $\\geq2$ (dotted)\ncollective glassy levels'); axs[0].set_ylim(-0.02, 1.02)
    axs[1].set_title('(b) split variance of the first glassy split\n$\\Delta_1/N$ (median, accepted or not)')
    axs[2].set_title('(c) sector mode $\\lambda_1/N$ (median)')
    axs[3].set_title('(d) window ratios: level 1 / sector (solid),\nlevel 2 / level 1 (dotted)'); axs[3].set_yscale('log')
    for a in axs: a.set_xlabel('coupling $\\beta_J$')
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'Fig2_scan.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Fig2_scan.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    json.dump(summary, open(RES / 'scan_summary.json', 'w'), indent=1)
    return summary


def law_fig():
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.5)); rows = []
    for f in glob.glob(str(RES / '*.json')):
        if 'summary' in f: continue
        o = json.load(open(f))
        if 'channels' not in o or 'anatomy' in o['name']: continue
        fam = o['name'].split('_')[0]; beta = float(o['name'].split('_b')[1].split('_s')[0])
        seed = int(o['name'].split('_s')[-1])
        if 'binarity' not in o and o['N'] <= 20:
            import importlib.util
            spec = importlib.util.spec_from_file_location('n1l', ROOT / 'experiments' / 'n1_levels.py'); n1l = importlib.util.module_from_spec(spec); spec.loader.exec_module(n1l)
            M = E.ExactModel(n1l.J_of(fam, o['N'], beta, seed)); v = np.linalg.eigh(M.C0)[1][:, -1]
            o['binarity'] = float(M.lam[0] / (M.p * np.abs(M.dot_grid(v))).sum() ** 2); json.dump(o, open(f, 'w'), indent=1)
        for l in range(len(o['n_classes'])):
            th = o['half_gamma'][l]
            if th is None or not o['channels'][l]: continue
            gg = np.geomspace(1e-4, 50, 800); pcurve = pred_curve(gg, o['channels'][l]); tp = float(gg[np.argmax(pcurve >= 0.5)])
            kap = float(np.exp(np.average(np.log([c['kappa'] for c in o['channels'][l]]), weights=[c['pi'] for c in o['channels'][l]])))
            rows.append(dict(level=l, N=o['N'], meas=th, pred=tp, fam=fam, beta=beta, kappa=kap, frac=o['frac'][l], g=o['gamma'], b=o.get('binarity', np.nan)))
    glass = lambda x: x['fam'] in ('bethe', 'sk') and x['beta'] >= 1.0
    ax = axs[0]
    for l in range(4):
        r = [x for x in rows if x['level'] == l and glass(x)]
        if r: ax.loglog([x['pred'] for x in r], [x['meas'] for x in r], 'o', ms=2.3, alpha=.45, color=LC[l], label=LN[l] + f' ({len(r)})')
    r = [x for x in rows if not glass(x)]
    ax.loglog([x['pred'] for x in r], [x['meas'] for x in r], 'x', ms=2.5, alpha=.35, color='0.55', label=f'$\\beta_J=0.5$ and controls ({len(r)})')
    lim = [3e-2, 2]; ax.set_xlim(lim); ax.set_ylim(lim); ax.plot(lim, lim, 'k-', lw=.7); ax.fill_between(lim, [x / 1.3 for x in lim], [x * 1.3 for x in lim], color='0.9', zorder=0)
    ax.set_xlabel('predicted $\\gamma_{1/2}$ (binary channels, $\\lambda_P$)'); ax.set_ylabel('measured $\\gamma_{1/2}$ (exact)')
    ax.set_title('(a) all levels of all exact instances'); ax.legend(frameon=False, fontsize=6.5)
    for k, l in enumerate([0, 1]):
        ax = axs[1 + k]; ss = np.geomspace(1e-2, 1e2, 200)
        for x in [x for x in rows if x['level'] == l and glass(x)][:500]:
            ax.plot(np.array(x['g'][1:]) * x['kappa'], np.array(x['frac'][1:]), color=LC[l], lw=.3, alpha=.2)
        ax.plot(ss, Ibin(ss), 'k-', lw=1.5, label='$I_{\\rm bin}(s)/\\log 2$'); ax.set_xscale('log'); ax.set_xlim(1e-2, 1e2)
        ax.set_xlabel('$s=\\gamma\\lambda_P$'); ax.set_title(f'({"bc"[k]}) collapse: {LN[l]}, $\\beta_J\\geq1$'); ax.legend(frameon=False, loc='upper left')
    ax = axs[3]
    for fam, mk, col in [('bethe', 'o', '#1f4e79'), ('sk', '^', '#c55a11'), ('ferro', 'x', '0.5'), ('mattis', 's', '#2e8b57')]:
        r = [x for x in rows if x['level'] == 0 and x['fam'] == fam and np.isfinite(x['b'])]
        if r: ax.semilogy([x['b'] for x in r], [x['meas'] / x['pred'] for x in r], mk, ms=2.5, alpha=.5, color=col, label=fam)
    ax.axhline(1, color='k', lw=.7); ax.axvline(np.pi / 2, color='0.6', ls=':', lw=.8); ax.text(np.pi / 2, 0.97, ' Gaussian', fontsize=7, color='0.4', transform=ax.get_xaxis_transform(), va='top')
    ax.set_xlabel('binarity of the sector mode $\\lambda_1/(\\mathbb{E}|v_1\\cdot Y|)^2$'); ax.set_ylabel('sector: measured / predicted')
    ax.set_title('(d) the prediction needs a binary mode'); ax.legend(frameon=False, fontsize=7)
    ax.set_yticks([0.8, 1, 1.5, 2, 3]); ax.set_yticklabels(['0.8', '1', '1.5', '2', '3']); ax.minorticks_off()
    fig.tight_layout(); fig.savefig(FIG / 'Fig06_law.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Fig06_law.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    st = {}
    groups = {'glass_b>=1': glass, 'glass_b0.5': lambda x: x['fam'] in ('bethe', 'sk') and x['beta'] < 1,
              'ferro': lambda x: x['fam'] == 'ferro', 'mattis': lambda x: x['fam'] == 'mattis',
              'bethe_b>=1': lambda x: x['fam'] == 'bethe' and x['beta'] >= 1.0, 'sk_b>=1': lambda x: x['fam'] == 'sk' and x['beta'] >= 1.0}
    for gname, gf in groups.items():
        for l in range(4):
            r = np.array([x['meas'] / x['pred'] for x in rows if x['level'] == l and gf(x)])
            if len(r): st[f'{gname}_L{l}'] = dict(n=len(r), median=float(np.median(r)), q25=float(np.quantile(r, .25)), q75=float(np.quantile(r, .75)),
                                                within30=float(np.mean(np.abs(np.log(r)) < np.log(1.3))), within50=float(np.mean(np.abs(np.log(r)) < np.log(1.5))))
    json.dump(st, open(RES / 'law_summary.json', 'w'), indent=1)
    return st


def _main():
    what = sys.argv[1:] or ['anatomy', 'temp', 'size', 'law']
    if 'temp' in what: temp_fig()
    if 'size' in what: print(json.dumps(size_fig())[:2000])
    if 'anatomy' in what:
        anatomy()

    if 'scan' in what: print(json.dumps(scan_fig(), indent=0)[:3000])
    if 'law' in what: print(law_fig())


# ------------------------------------------------------------------ second round (2026-10-31): temperature and size
RS = ROOT / 'results' / 'N1S'
BGRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def load_struct(fam, N, b, s):
    for d in (RS, RES):
        f = d / f'{fam}_N{N}_b{b}_s{s}.json'
        if f.exists(): return json.load(open(f))
    return None


def first_contrast(o):
    if not o or not o.get('contrasts') or not o['contrasts'] or not o['contrasts'][0]: return None
    return np.array(o['contrasts'][0][0]['d'])


def draw_tree(ax, o, title):
    ax.axis('off'); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.15, 1.12); ax.set_title(title, fontsize=8.5)
    L = o['n_levels_glassy']; lv = tree_nodes(o['level_info'], L); X = [None] * len(lv)
    n_last = len(lv[-1]); X[-1] = list(np.linspace(0.08, 0.92, n_last)) if n_last > 1 else [0.5]
    for l in range(len(lv) - 2, -1, -1):
        X[l] = [float(np.mean([X[l + 1][j] for j, (_, p) in enumerate(lv[l + 1]) if p == c])) for c in range(len(lv[l]))]
    dy = 0.95 / 3
    for l, nodes in enumerate(lv):
        y = 1.0 - dy * l
        for j, (m, par) in enumerate(nodes):
            carried = par is not None and sum(1 for _, p in nodes if p == par) == 1
            if par is not None: ax.plot([X[l - 1][par], X[l][j]], [y + dy, y], color='0.65', lw=0.8, zorder=0, ls=':' if carried else '-')
            col = LC[min(l, 3)] if l > 0 else '0.35'
            if carried: ax.scatter(X[l][j], y, s=420 * m + 10, facecolor='white', edgecolor=col, lw=1.1, zorder=2)
            else: ax.scatter(X[l][j], y, s=420 * m + 10, color=col, zorder=2, edgecolor='white', lw=.5)
            if l > 0: ax.text(X[l][j], y - 0.1, f'{m:.2f}', ha='center', fontsize=5.8)
    if L == 0: ax.text(0.5, 0.55, 'no collective split\nbelow the sector', ha='center', fontsize=7, color='0.4')


def temp_fig(N=24, seed=2, show=(1.0, 1.5, 2.0, 2.5)):
    fig = plt.figure(figsize=(13, 5.8)); gs = fig.add_gridspec(2, 4, hspace=.35, wspace=.32, height_ratios=[0.75, 1])
    for k, b in enumerate(show):
        o = load_struct('bethe', N, b, seed); draw_tree(fig.add_subplot(gs[0, k]), o, f'({"abcd"[k]}) $\\beta_J={b}$: classes inside one sector')
    bs = [b for b in BGRID if load_struct('bethe', N, b, seed)]
    O = [load_struct('bethe', N, b, seed) for b in bs]
    ax = fig.add_subplot(gs[1, 0]); ax.set_title(f'(e) split variances, $N={N}$ instance')
    ax.semilogy(bs, [o['lam'][0] for o in O], 'k-o', ms=3, label='sector ($\\lambda_1$)')
    for o, b in zip(O, bs):
        for l, sp in enumerate(o['level_info']):
            for s_ in sp: ax.semilogy(b, s_['kappa'], 'o', ms=4, color=LC[l + 1])
    for l in range(1, 4): ax.plot([], [], 'o', color=LC[l], label=LN[l])
    ax.axhline(2, color='0.5', ls=':', lw=.8); ax.text(bs[0], 2.1, 'gate $\\Delta=2$', fontsize=6.5, color='0.4')
    ax.set_xlabel('$\\beta_J$'); ax.set_ylabel('split variance $\\Delta$ of accepted splits'); ax.legend(frameon=False, fontsize=6.5)
    ax.set_yticks([2, 3, 5, 10, 15]); ax.set_yticklabels(['2', '3', '5', '10', '15']); ax.minorticks_off()
    ax = fig.add_subplot(gs[1, 1]); ax.set_title('(f) predicted decision times, same instance')
    for l in range(4):
        xs_, ys_ = [], []
        for o, b in zip(O, bs):
            if l < len(o['pred_half']): xs_.append(b); ys_.append(o['pred_half'][l])
        if xs_: ax.semilogy(xs_, ys_, '-o', ms=3, color=LC[l], label=LN[l])
    ax.set_xlabel('$\\beta_J$'); ax.set_ylabel('$\\gamma_{1/2}$ from $\\lambda_P$'); ax.legend(frameon=False, fontsize=6.5)
    ax.set_yticks([0.07, 0.1, 0.15, 0.2, 0.3, 0.4]); ax.set_yticklabels(['0.07', '0.1', '0.15', '0.2', '0.3', '0.4']); ax.minorticks_off()
    ax = fig.add_subplot(gs[1, 2]); ax.set_title('(g) instances with $\\geq1$ (solid), $\\geq2$ (dotted)\ncollective glassy levels')
    for fam, NN, S, mk, col in [('bethe', 16, 60, 'o', '#1f4e79'), ('bethe', 20, 24, 's', '#2e75b6'), ('bethe', 24, 12, 'D', '#9dc3e6'), ('ferro', 16, 30, 'x', '0.45')]:
        xb, f1, f2 = [], [], []
        for b in BGRID:
            Os = [load_struct(fam, NN, b, s) for s in range(1, S + 1)]; Os = [x for x in Os if x]
            if len(Os) < 0.8 * S: continue
            xb.append(b); f1.append(np.mean([x['n_levels_glassy'] >= 1 for x in Os])); f2.append(np.mean([x['n_levels_glassy'] >= 2 for x in Os]))
        lab = f'{"Bethe" if fam == "bethe" else "ferro (|J|)"} $N={NN}$'
        ax.plot(xb, f1, '-' + mk, color=col, ms=3.5, label=lab); ax.plot(xb, f2, ':' + mk, color=col, ms=2.5, mfc='none')
    ax.axvline(1.34, color='0.6', lw=.7, ls='--'); ax.set_xlabel('$\\beta_J$'); ax.set_ylim(-0.02, 1.02); ax.legend(frameon=False, fontsize=6.5)
    ax = fig.add_subplot(gs[1, 3]); ax.set_title('(h) same split at $\\beta_J$ and at $\\beta_J=2.5$?\nmedian $|\\cos|$ of first glassy contrasts')
    pers = {}
    for NN, S, mk, col in [(16, 60, 'o', '#1f4e79'), (20, 24, 's', '#2e75b6'), (24, 12, 'D', '#9dc3e6')]:
        xb, med, nn = [], [], []
        for b in BGRID:
            cs = []
            for s in range(1, S + 1):
                a_, r_ = first_contrast(load_struct('bethe', NN, b, s)), first_contrast(load_struct('bethe', NN, 2.5, s))
                if a_ is not None and r_ is not None: cs.append(abs(a_ @ r_))
            if len(cs) >= 3: xb.append(b); med.append(np.median(cs)); nn.append(len(cs))
        ax.plot(xb, med, '-' + mk, color=col, ms=3.5, label=f'Bethe $N={NN}$'); pers[NN] = dict(beta=xb, median=med, n=nn)
    ax.axvline(1.34, color='0.6', lw=.7, ls='--'); ax.set_ylim(0, 1.02); ax.set_xlabel('$\\beta_J$'); ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(FIG / 'Fig04_temperature.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Fig04_temperature.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    json.dump(dict(persistence=pers, seed_track={str(b): dict(levels=o['n_levels_glassy'], lam1=o['lam'][0], pred=o['pred_half'],
               kappas=[[s_['kappa'] for s_ in sp] for sp in o['level_info']], masses=[[s_['masses'] for s_ in sp] for sp in o['level_info']]) for o, b in zip(O, bs)}),
              open(RES / 'temperature_summary.json', 'w'), indent=1)


def size_fig():
    """Scan across couplings and sizes: strength of the first glassy split, sector mode, measured window ratios, sector-first fraction."""
    fams = [('bethe', 16, 60, 'o', '#1f4e79'), ('bethe', 20, 24, 's', '#2e75b6'), ('bethe', 24, 12, 'D', '#9dc3e6'),             ('ferro', 16, 30, 'x', '0.45'), ('ferro', 20, 12, '+', '0.65')]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3)); summ = {}
    for fam, N, S, mk, col in fams:
        bset = BGRID + ([] if fam != 'sk' else []); xb, k1, l1, r10, r21, sf, nd = [], [], [], [], [], [], []
        for b in sorted(set(BGRID) | {3.0}):
            Os = [load_struct(fam, N, b, s) for s in range(1, S + 1)]; Os = [x for x in Os if x]
            if len(Os) < 6: continue
            kk = [[c['kappa'] for c in x['candidates'] if c['level'] == 1] for x in Os]
            xb.append(b); k1.append(np.median([k[0] / N for k in kk if k])); l1.append(np.median([x['lam'][0] / N for x in Os]))
            D = [json.load(open(RES / f'{fam}_N{N}_b{b}_s{s}.json')) for s in range(1, S + 1) if (RES / f'{fam}_N{N}_b{b}_s{s}.json').exists()]
            rr = [x['half_gamma'][1] / x['half_gamma'][0] for x in D if x['n_levels_glassy'] >= 1 and x['half_gamma'][1]]
            rr2 = [x['half_gamma'][2] / x['half_gamma'][1] for x in D if x['n_levels_glassy'] >= 2 and x['half_gamma'][2]]
            ss = [x['half_gamma'][0] <= min(v for v in x['half_gamma'] if v) for x in D if x['n_levels_glassy'] >= 1]
            r10.append(np.median(rr) if len(rr) >= 3 else np.nan); r21.append(np.median(rr2) if len(rr2) >= 3 else np.nan)
            sf.append(np.mean(ss) if len(ss) >= 3 else np.nan); nd.append(len(D))
        lab = f'{dict(bethe="Bethe", sk="SK", ferro="ferro (|J|)")[fam]} $N={N}$'
        axs[0].plot(xb, k1, '-' + mk, color=col, ms=3.5, label=lab); axs[1].plot(xb, l1, '-' + mk, color=col, ms=3.5)
        xa = np.array(xb, float)
        for arr, ax_, st in [(np.array(r10, float), axs[2], '-'), (np.array(r21, float), axs[2], ':'), (np.array(sf, float), axs[3], '-')]:
            m_ = np.isfinite(arr)
            ax_.plot(xa[m_], arr[m_], st + mk, color=col, ms=3.5 if st == '-' else 2.5, mfc=None if st == '-' else 'none')
        summ[f'{fam}_N{N}'] = dict(beta=xb, kappa1_over_N=k1, lam1_over_N=l1, ratio_l1_sector=r10, ratio_l2_l1=r21, sector_first=sf, n_dyn=nd)
    axs[0].set_title('(a) split variance of the best first glassy\nsplit $\\Delta_1/N$ (median, accepted or not)')
    axs[1].set_title('(b) sector mode $\\lambda_1/N$ (median)')
    axs[2].set_title('(c) measured decision-time ratios: glassy 1 / sector\n(solid), glassy 2 / glassy 1 (dotted)'); axs[2].set_yscale('log'); axs[2].axhline(1, color='0.6', lw=.7)
    axs[2].set_yticks([0.8, 1, 1.5, 2, 3, 4]); axs[2].set_yticklabels(['0.8', '1', '1.5', '2', '3', '4']); axs[2].minorticks_off()
    axs[3].set_title('(d) fraction of instances (with a glassy\nlevel) in which the sector is decided first'); axs[3].set_ylim(0, 1.05)
    for a in axs: a.set_xlabel('coupling $\\beta_J$')
    axs[1].legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=6.5, ncol=1, loc='lower right')
    fig.tight_layout(); fig.savefig(FIG / 'Fig05_sizes.pdf', bbox_inches='tight'); fig.savefig(FIG / 'Fig05_sizes.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    json.dump(summ, open(RES / 'size_summary.json', 'w'), indent=1); return summ


if __name__ == '__main__':
    _main()
