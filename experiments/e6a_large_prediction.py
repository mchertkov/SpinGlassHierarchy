"""Stage 6A (prediction use) at N=32/64: decision clocks from equilibrium class geometry alone.

The centroid channel replaces each frozen leaf by its equilibrium mean magnetisation and
propagates it through the SAME Gaussian observation channel.  Inputs: leaf masses and leaf
means estimated from equilibrium samples (Stage-6 PT libraries for Bethe; exact samples for
nulls).  No posterior sampling.  Predictions are compared with the frozen Stage 5D/5E
measured half-information times (first grid time reaching half the gamma=3.2 information).
"""
import json
import numpy as np
from e6_common import (ROOT, OUT, load_library, frozen_large_tree, frozen_null_tree, null_sample,
                       hierarchy_geometry, json_dump)
from e6a_contrast_algorithms import centroid_channel, contrast_norms, half_time
from e5b_sequential_smc import GRID as G5

FINE = np.r_[0., np.geomspace(1e-3, 5., 56)]


def quantized_half(curve, grid, horizon=3.2):
    k = np.searchsorted(grid, horizon - 1e-12); target = .5 * curve[k]
    ix = np.flatnonzero(curve[:k + 1] >= target)
    return float(grid[ix[0]]) if len(ix) and curve[k] > 0 else None


def predict_from(pi, means, seed):
    cfine = centroid_channel(pi, means, FINE, seed=seed)
    cq = centroid_channel(pi, means, G5, seed=seed)
    return dict(pred_half_continuous=[half_time(cfine[:, l], FINE, 3.2) for l in range(2)],
                pred_half_grid=[quantized_half(cq[:, l], G5) for l in range(2)],
                curve_fine=cfine, D2=list(contrast_norms(pi, means)))


def main():
    rows = []
    d5 = json.loads((ROOT / 'results' / 'E5D' / 'STAGE5D_METRICS.json').read_text())
    meas = {(g['N'], g['seed']): g['half_times_accepted'] for g in d5['groups']}
    for N in (32, 64):
        for d in (4, 5, 8):
            Y, _ = load_library(N, d); tree = frozen_large_tree(N, d)
            geo = hierarchy_geometry(Y, tree)
            r = predict_from(geo['pi'], geo['means'], seed=100 * N + d)
            rows.append(dict(model='bethe', N=N, ident=d, measured_half=meas[(N, d)], **r))
    e = json.loads((ROOT / 'results' / 'E5E' / 'STAGE5E_RESULTS.json').read_text())
    for row in e['rows']:
        if row['model'] not in ('two_state', 'product'): continue
        N, s = row['N'], row['seed']
        tree = frozen_null_tree(row['model'], N, s)
        Y = null_sample(row['model'], 200000, N, np.random.default_rng(5 + N + s))
        geo = hierarchy_geometry(Y, tree)
        r = predict_from(geo['pi'], geo['means'], seed=7 * N + s)
        rows.append(dict(model=row['model'], N=N, ident=s, measured_half=[row['half_coarse'], row['half_fine']], **r))
    for r in rows:
        print(r['model'], r['N'], r['ident'], 'measured', r['measured_half'], 'pred grid', r['pred_half_grid'],
              'pred cont', np.round(r['pred_half_continuous'], 3), 'D2', np.round(r['D2'], 2))
    json_dump(OUT / 'E6A' / 'large_prediction.json', dict(grid=FINE, rows=rows,
              note='Centroid-channel prediction from equilibrium leaf means/masses only; measured = Stage 5D (accepted paths) / Stage 5E.'))


if __name__ == '__main__':
    main()
