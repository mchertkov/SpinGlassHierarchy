"""Stage 6: large equilibrium libraries for the six frozen N=32/64 Bethe disorders.

Heat-bath parallel tempering (28 inverse temperatures, same ladder as Stage 4D-H/5),
several independently seeded histories per disorder, cold samples recorded after burn-in.
Each recorded cold configuration is multiplied by an independent random global sign: the
zero-field Gibbs law is exactly Z2 symmetric, so this is an exact symmetrisation, not a
mixing claim.  Libraries are used (a) to train learned denoisers and (b) for class means.
Validation against the frozen Stage-5 libraries is reported in the JSON summary.
"""
from __future__ import annotations
import argparse, time
import numpy as np
from numba import njit
from e6_common import P, OUT, BETA_J, LARGE_DISORDERS, frozen_large_tree, predict, json_dump, library_path, adjusted_rand


@njit(cache=True)
def _pt(X, nbr, Jn, deg, betas, sweeps, burn, stride, seed, out, tagtrack):
    np.random.seed(seed)
    R, N = X.shape
    perm = np.arange(R)          # perm[k] = replica index currently at temperature k
    nsave = 0
    acc = np.zeros(R - 1); att = np.zeros(R - 1)
    trips = np.zeros(R, np.int64); state = np.zeros(R, np.int64)  # 0 unknown, 1 last visited hot, 2 last visited cold
    for sw in range(sweeps):
        for k in range(R):
            r = perm[k]; b = betas[k]
            for i in range(N):
                loc = 0.
                for a in range(deg[i]):
                    loc += Jn[i, a] * X[r, nbr[i, a]]
                p = 0.5 * (1. + np.tanh(b * loc))
                X[r, i] = 1 if np.random.random() < p else -1
        # replica exchange, alternating even/odd pairs
        start = sw % 2
        for k in range(start, R - 1, 2):
            r1 = perm[k]; r2 = perm[k + 1]
            e1 = 0.; e2 = 0.
            for i in range(N):
                for a in range(deg[i]):
                    j = nbr[i, a]
                    if j > i:
                        e1 += Jn[i, a] * X[r1, i] * X[r1, j]
                        e2 += Jn[i, a] * X[r2, i] * X[r2, j]
            # weights exp(beta*E); swap acceptance min(1, exp((b_k-b_{k+1})(E2-E1)))
            dl = (betas[k] - betas[k + 1]) * (e2 - e1)
            att[k] += 1
            if dl >= 0 or np.random.random() < np.exp(dl):
                perm[k] = r2; perm[k + 1] = r1; acc[k] += 1
        # round trips of replicas: hot end k=0, cold end k=R-1
        r = perm[0]
        if state[r] == 2: trips[r] += 1
        state[r] = 1
        r = perm[R - 1]
        if state[r] == 0: state[r] = 2
        elif state[r] == 1: state[r] = 2
        if sw >= burn and (sw - burn) % stride == 0 and nsave < out.shape[0]:
            out[nsave] = X[perm[R - 1]]
            tagtrack[nsave] = perm[R - 1]
            nsave += 1
    return nsave, acc / np.maximum(att, 1), trips


def run(N, disorder, histories=4, sweeps=220000, burn=20000, stride=5):
    inst = P.bethe_spin_glass(N, 3, BETA_J, 'gauss', disorder)
    betas = BETA_J * np.linspace(1e-6, 1., 28) ** 1.15
    deg = np.bincount(inst.edges.ravel(), minlength=N).astype(np.int64)
    nbr = inst.nbr.astype(np.int64); Jn = inst.Jn.astype(np.float64)
    allY = []; diag = []
    t0 = time.monotonic()
    for hist in range(histories):
        rng = np.random.default_rng(61800 + 1000 * N + 37 * disorder + hist)
        X = rng.choice(np.array([-1, 1], np.int8), size=(len(betas), N)).astype(np.int8)
        nout = (sweeps - burn) // stride
        out = np.zeros((nout, N), np.int8); tag = np.zeros(nout, np.int64)
        ns, accr, trips = _pt(X, nbr, Jn, deg, betas, sweeps, burn, stride, int(rng.integers(2**31 - 1)), out, tag)
        Y = out[:ns].astype(np.int8)
        signs = rng.choice(np.array([-1, 1], np.int8), size=(ns, 1))
        allY.append(Y * signs)
        e = inst.energy(Y.astype(float))
        diag.append(dict(history=hist, samples=int(ns), min_swap_acceptance=float(accr.min()),
                         roundtrips_total=int(trips.sum()), roundtrips_min=int(trips.min()),
                         mean_beta_energy=float(BETA_J * e.mean()), energy_first_half=float(e[:ns // 2].mean()),
                         energy_second_half=float(e[ns // 2:].mean())))
    Y = np.concatenate(allY)
    # ---- validation against the frozen Stage-5 tree and library
    tree = frozen_large_tree(N, disorder)
    from e5_largeN_importance import load
    a, _ = load(N, disorder, 0); b, _ = load(N, disorder, 1)
    frozen = np.r_[a.reshape(-1, N), b.reshape(-1, N)].astype(float)
    frozen_sym = np.r_[frozen, -frozen]
    _, lf_new = predict(Y.astype(float), tree); _, lf_old = predict(frozen_sym, tree)
    pi_new = np.bincount(lf_new, minlength=4) / len(lf_new); pi_old = np.bincount(lf_old, minlength=4) / len(lf_old)
    per_hist = []
    for k in range(histories):
        _, l = predict(allY[k].astype(float), tree); per_hist.append((np.bincount(l, minlength=4) / len(l)).tolist())
    e_new = inst.energy(Y.astype(float)); e_old = inst.energy(frozen)
    # second, independently trained tree on the new library vs frozen tree (held-out ARI)
    from e3b_e1_heldout_hierarchy import train_tree
    half = len(Y) // 2
    tnew = train_tree(np.r_[Y[:half:4], -Y[:half:4]].astype(float))
    hold = Y[half::7].astype(float)
    c1, f1 = predict(hold, tree); c2, f2 = predict(hold, tnew)
    ari = [adjusted_rand(c1, c2), adjusted_rand(f1, f2)]
    summary = dict(N=N, disorder=disorder, betas=betas, histories=diag, total_samples=len(Y), sweeps=sweeps, burn=burn, stride=stride,
                   leaf_mass_new=pi_new, leaf_mass_frozen_symmetrised=pi_old, leaf_mass_per_history=per_hist,
                   max_leaf_mass_diff_vs_frozen=float(np.abs(pi_new - pi_old).max()),
                   max_leaf_mass_spread_histories=float(np.ptp(np.array(per_hist), axis=0).max()),
                   mean_energy_new=float(e_new.mean()), mean_energy_frozen=float(e_old.mean()),
                   energy_se_frozen=float(e_old.std() / np.sqrt(len(e_old))),
                   ari_frozen_tree_vs_retrained_on_new_library=ari,
                   seconds=round(time.monotonic() - t0, 1),
                   note='Recorded configurations are temporally correlated within each history; global signs randomised exactly.')
    path = library_path(N, disorder); path.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(path, Y=Y, history=np.repeat(np.arange(histories), [len(z) for z in allY]))
    json_dump(path.with_suffix('.json'), summary)
    print(N, disorder, 'samples', len(Y), 'pi new', np.round(pi_new, 3), 'pi frozen', np.round(pi_old, 3),
          'E', round(e_new.mean(), 3), round(e_old.mean(), 3), 'trips', [d['roundtrips_min'] for d in diag],
          'ARI', np.round(ari, 3), 's', summary['seconds'], flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--N', type=int, nargs='*', default=[32, 64])
    p.add_argument('--disorders', type=int, nargs='*', default=list(LARGE_DISORDERS))
    p.add_argument('--sweeps', type=int, default=220000)
    a = p.parse_args()
    for N in a.N:
        for d in a.disorders: run(N, d, sweeps=a.sweeps)
