"""Stage 7, experiment 1: static predictors and dynamic detectors of decisions on the frozen exact N=16 cohort
(12 Bethe disorders, 5 product nulls, 5 two-state nulls; results/E5F/exact16).  Uses the stored planted fields H
(gamma grid of Stage 5) and the stored measured half-information times of the predeclared coarse/fine classes.
Static: C0 spectrum, <Q^2>_J, spectral windows 1/lambda_k, Gaussian surrogate.  Dynamic (exact, same paths):
chi_SG per path, mean log-rate gamma*E chi, Var chi, std of per-path log-rate, excess over the Gaussian surrogate.
Usage: python3 experiments/s7_exact16_static.py [kind ...]   -> results/S7/exact16_<file>.json"""
import sys, json, glob, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'experiments')]
import e5_exact_transients as E
OUT = ROOT / 'results' / 'S7'; OUT.mkdir(parents=True, exist_ok=True)


def gaussian_surrogate(C0, gam):
    lam = np.clip(np.linalg.eigvalsh(C0), 1e-12, None); N = len(lam)
    post = lam[None] / (1 + gam[:, None] * lam[None]); return (lam.sum() - post.sum(1)) / N, (post ** 2).sum(1) / N


def run(f):
    t0 = time.time(); d = json.load(open(f)); z = np.load(f.replace('.json', '.npz'))
    S = E.setup(d['kind'], d['seed']); X = S.X.astype(np.float32); mu = np.exp(S.logmu); mu /= mu.sum(); N = 16
    C0 = (X.T.astype(float) * mu) @ X.astype(float); m0 = mu @ X.astype(float); C0 = C0 - np.outer(m0, m0)
    lam = np.sort(np.linalg.eigvalsh(C0))[::-1]; Q2 = float(np.trace(C0 @ C0) / N ** 2)
    gam = z['gamma']; H = z['H']; npth = H.shape[1]; iu = np.triu_indices(N, 1); Pp = (X[:, iu[0]] * X[:, iu[1]])
    chi = np.zeros((len(gam), npth)); q = np.zeros((len(gam), npth))
    for k in range(len(gam)):
        lp = S.logmu[None] + H[k].astype(np.float32) @ X.T; w = np.exp(lp - logsumexp(lp, 1, keepdims=True)).astype(np.float32)
        m = w @ X; off = w @ Pp - m[:, iu[0]] * m[:, iu[1]]
        chi[k] = (((1 - m ** 2) ** 2).sum(1) + 2 * (off ** 2).sum(1)) / N; q[k] = (m ** 2).sum(1) / N
    qG, chiG = gaussian_surrogate(C0, gam); pos = gam > 0; g = gam[pos]
    det = dict(lograte=(g * chi[pos].mean(1)), var_chi=chi[pos].var(1), std_lograte=(g[:, None] * chi[pos]).std(1),
               excess=chi[pos].mean(1) - chiG[pos], excess_rel=(chi[pos].mean(1) - chiG[pos]) / chiG[pos])
    peaks = {k: float(g[int(np.argmax(v))]) for k, v in det.items()}
    exr = np.where(det['excess'] > 0.05 * det['excess'].max())[0]
    res = dict(file=Path(f).name, kind=d['kind'], seed=d['seed'], accepted=bool(d.get('accepted_candidate_tree')), heldout_ARI=d['heldout_ARI'],
               measured_half=d['half_information_times_gamma'], lam=lam.tolist(), Q2=Q2, chi0=float(N * Q2), chi_at_gamma0=float(chi[0].mean()),
               spectral=(1 / lam[:4]).tolist(), gap12=float(lam[0] / lam[1]), gap23=float(lam[1] / lam[2]),
               lam2_over_bulk=float(lam[1] / np.median(lam[2:])), gamma=g.tolist(), detectors={k: v.tolist() for k, v in det.items()},
               peaks=peaks, excess_positive_range=[float(g[exr.min()]), float(g[exr.max()])] if len(exr) else None,
               q_mean=q[pos].mean(1).tolist(), q_gauss=qG[pos].tolist(), chi_mean=chi[pos].mean(1).tolist(), chi_gauss=chiG[pos].tolist(),
               seconds=time.time() - t0)
    (OUT / f"exact16_{Path(f).stem}.json").write_text(json.dumps(res))
    print(f"{Path(f).stem:24s} acc={res['accepted']!s:5} half={np.round(res['measured_half'], 3).tolist()} 1/lam={np.round(res['spectral'][:2], 3).tolist()} "
          f"gap12={res['gap12']:.1f} gap23={res['gap23']:.1f} Q2={Q2:.2f} chi(0)={res['chi_at_gamma0']:.2f}/{res['chi0']:.2f} "
          f"peaks lr={peaks['lograte']:.3f} var={peaks['var_chi']:.3f} std={peaks['std_lograte']:.3f} exc={peaks['excess']:.3f} {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    kinds = sys.argv[1:] or ['bethe', 'product', 'two_state']
    for f in sorted(glob.glob(str(ROOT / 'results/E5F/exact16/*.json'))):
        if json.load(open(f))['kind'] in kinds and not (OUT / f"exact16_{Path(f).stem}.json").exists(): run(f)
