"""Partial commitments of single trajectories (N=24 instance of the anatomy figure).
Panel data: for chosen trajectories, the posterior distribution over the finest classes (both sectors) along gamma, and the conditional
partial commitments q_l(gamma) = p_{c_l(Y)} / p_{c_{l-1}(Y)}, the conditional partial commitment to the endpoint's class c_l(Y).
usage: python experiments/n3_partial.py   -> results/N3/partial_N24.npz
"""
from pathlib import Path
import sys, json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'experiments')]
import exacttree as E
OUT = ROOT / 'results' / 'N3'; OUT.mkdir(parents=True, exist_ok=True)

name = 'bethe_N24_b2.5_s2_anatomy'
z = np.load(ROOT / 'results' / 'N1' / f'{name}.npz')
M = E.ExactModel(E.bethe_J(24, 2.5, 2)); T = E.build_tree(M)
labs, Ks = T['labels'], T['n_classes']; L = len(labs)
fin = labs[-1].ravel()
rep = np.array([np.argmax(fin == c) for c in range(Ks[-1])])
maps = np.array([lab.ravel()[rep] for lab in labs])           # (L, K_finest): class of each finest class at every level
prior = np.bincount(fin, weights=M.p.ravel(), minlength=Ks[-1])
np.savez_compressed(OUT / 'partial_N24.npz', gamma=z['gamma'], post_fine=z[f'post{L-1}'],
                    **{f'post{l}': z[f'post{l}'] for l in range(L)}, **{f'plant{l}': z[f'plant{l}'] for l in range(L)},
                    maps=maps, prior=prior, w=z['w'])
print('saved', maps.shape, Ks)
