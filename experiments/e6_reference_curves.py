"""Collect reference class-information curves (coarse, fine|coarse) for every Stage-6 target.

N=16 Bethe : exact enumeration, Stage 6A (1536 planted paths, 57-point grid).
N=32/64    : Stage 5D corrected posterior class masses on the accepted paths (frozen arrays;
             e5d_analyze.reconstruct + make_metrics), 21-point grid, horizon 3.2.
Nulls 32/64: Stage 5E exact-posterior-sampling curves for training replicate 4.
"""
import json
import numpy as np
from e6_common import ROOT, OUT, json_dump, N16_ACCEPTED, N16_REJECTED


def main():
    ref = {}
    for s in N16_ACCEPTED + N16_REJECTED:
        f = OUT / 'E6A' / f'bethe_seed{s}.json'
        if not f.exists(): continue
        d = json.loads(f.read_text())
        ref[f'bethe16_N16_id{s}'] = dict(gamma=d['grid'], info=d['I_contrast'], source='exact N16 (Stage 6A)')
    from e5d_analyze import reconstruct, make_metrics
    from e5_largeN_importance import load
    from e3b_e1_heldout_hierarchy import train_tree
    E5B = ROOT / 'results' / 'E5B'
    for n in (32, 64):
        for seed in (4, 5, 8):
            g, Y, m, err, repaired, unresolved, counts, pt_gates, orig = reconstruct(n, seed)
            a, _ = load(n, seed, 0)
            tree = train_tree(np.r_[a[:, :8].reshape(-1, n), -a[:, :8].reshape(-1, n)])
            pi = np.asarray(json.loads((E5B / f'N{n}_bethe_seed{seed}_p24_P768.json').read_text())['prior_leaf_mass'])
            valid = ~unresolved
            _, cur, half = make_metrics(g, Y[valid], m[valid], pi, tree, err[valid], np.ones(valid.sum(), bool))
            ref[f'bethe_N{n}_id{seed}'] = dict(gamma=g.tolist(), info=cur.tolist(), half=half, source='Stage 5D corrected SMC, accepted paths')
    for model in ('two_state', 'product'):
        for n in (32, 64):
            d = json.loads((ROOT / 'results' / 'E5E' / f'{model}_N{n}_rep4.json').read_text())
            ref[f'{model}_N{n}_id4'] = dict(gamma=d['gamma'], info=d['information_curves'], half=d['information_half_times'],
                                            source='Stage 5E exact posterior sampling')
    json_dump(OUT / 'reference_curves.json', ref)
    for k, v in ref.items(): print(k, np.round(np.array(v['info'])[-1], 3), v.get('half'))


if __name__ == '__main__':
    main()
