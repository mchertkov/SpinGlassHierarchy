"""Stage 6B: train small learned denoisers m_theta(h, gamma) ~ E[Y | H_gamma = h] on frozen targets.

Architecture (one hidden layer, chosen for unit-level interpretability):
    a_k   = tanh( w_k . h + c_k log(gamma) + b_k ),        k = 1..W   ("units")
    m_hat = tanh( V a + D * h + b ),                        D diagonal  (local skip path)
The network sees ONLY denoising pairs (Y, H_gamma = gamma Y + sqrt(gamma) Z) with Y drawn
from the equilibrium law.  It never sees a class label, a tree, or a contrast vector.
Loss: mean squared error (the Bayes-optimal minimiser is the posterior mean, Eq. yhat).
Gamma ~ log-uniform on [1e-3, 10].

Targets: exact N=16 Bethe (fresh exact samples), N=32/64 Bethe (Stage-6 PT libraries,
histories 0-2 for training, history 3 held out), and the two exactly-sampled nulls at N=32/64.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from e6_common import OUT, ExactHierarchy, load_library, null_sample

torch.set_num_threads(1)
GLO, GHI = 1e-3, 10.


class Denoiser(torch.nn.Module):
    def __init__(self, N, W):
        super().__init__()
        self.inp = torch.nn.Linear(N, W)
        self.cg = torch.nn.Parameter(torch.zeros(W))
        self.out = torch.nn.Linear(W, N)
        self.D = torch.nn.Parameter(torch.full((N,), .5))
        torch.nn.init.normal_(self.inp.weight, std=.3 / np.sqrt(N)); torch.nn.init.zeros_(self.inp.bias)
        torch.nn.init.normal_(self.out.weight, std=.3 / np.sqrt(W)); torch.nn.init.zeros_(self.out.bias)

    def hidden(self, h, lg):
        return torch.tanh(self.inp(h) + lg[:, None] * self.cg[None, :])

    def forward(self, h, lg, return_hidden=False):
        a = self.hidden(h, lg)
        m = torch.tanh(self.out(a) + self.D[None, :] * h)
        return (m, a) if return_hidden else m


class Source:
    """Endpoint sampler: exact N=16, PT library, or exact null."""

    def __init__(self, kind, N, ident):
        self.kind, self.N, self.ident = kind, N, ident
        if kind == 'bethe16':
            self.E = ExactHierarchy('bethe', ident)
        elif kind == 'bethe':
            Y, z = load_library(N, ident); hist = z['history']
            self.train = Y[hist < 3]; self.test = Y[hist == 3]
        elif kind in ('two_state', 'product'):
            pass
        else:
            raise ValueError(kind)

    def draw(self, n, rng, split='train'):
        if self.kind == 'bethe16': return self.E.sample_prior(n, rng)
        if self.kind == 'bethe':
            pool = self.train if split == 'train' else self.test
            return pool[rng.integers(0, len(pool), n)]
        return null_sample(self.kind, n, self.N, rng)


def tag(kind, N, ident, W, seed):
    return f'{kind}_N{N}_id{ident}_W{W}_s{seed}'


def train(kind, N, ident, W, seed, steps=12000, batch=512, lr=3e-3, checkpoints=0):
    t0 = time.monotonic()
    torch.manual_seed(1000 + seed); rng = np.random.default_rng(777000 + 97 * seed + 13 * W + 7 * ident + N)
    src = Source(kind, N, ident)
    net = Denoiser(N, W)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=.05)
    ck_steps = set(np.unique(np.geomspace(10, steps, checkpoints).astype(int)).tolist()) if checkpoints else set()
    ck = {}
    hist = []
    for it in range(1, steps + 1):
        Y = src.draw(batch, rng)
        g = np.exp(rng.uniform(np.log(GLO), np.log(GHI), batch))
        H = g[:, None] * Y + np.sqrt(g)[:, None] * rng.standard_normal(Y.shape)
        Yt = torch.as_tensor(Y, dtype=torch.float32); Ht = torch.as_tensor(H, dtype=torch.float32)
        lg = torch.as_tensor(np.log(g), dtype=torch.float32)
        loss = ((net(Ht, lg) - Yt) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 200 == 0: hist.append([it, float(loss.detach())])
        if it in ck_steps:
            ck[it] = {k: v.detach().clone().numpy() for k, v in net.state_dict().items()}
    d = OUT / 'E6B' / 'models'; d.mkdir(parents=True, exist_ok=True)
    name = tag(kind, N, ident, W, seed)
    np.savez_compressed(d / f'{name}.npz', **{k: v.detach().numpy() for k, v in net.state_dict().items()})
    if ck:
        np.savez_compressed(d / f'{name}_ckpt.npz', steps=np.array(sorted(ck)),
                            **{f'{k}@{s}': v for s in sorted(ck) for k, v in ck[s].items()})
    (d / f'{name}.json').write_text(json.dumps(dict(kind=kind, N=N, ident=ident, W=W, seed=seed, steps=steps, batch=batch, lr=lr,
                                                    loss_history=hist, seconds=round(time.monotonic() - t0, 1))))
    print('TRAINED', name, 'final loss', round(hist[-1][1], 3), 's', round(time.monotonic() - t0, 1), flush=True)
    return net


def load_net(kind, N, ident, W, seed, step=None):
    d = OUT / 'E6B' / 'models'; name = tag(kind, N, ident, W, seed)
    net = Denoiser(N, W)
    if step is None:
        z = np.load(d / f'{name}.npz'); sd = {k: torch.as_tensor(z[k]) for k in z.files}
    else:
        z = np.load(d / f'{name}_ckpt.npz'); sd = {k.split('@')[0]: torch.as_tensor(z[k]) for k in z.files if k.endswith(f'@{step}')}
    net.load_state_dict(sd); net.eval()
    return net


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', required=True, help='JSON list of [kind,N,ident,W,seed,checkpoints]')
    p.add_argument('--steps', type=int, default=12000)
    a = p.parse_args()
    for kind, N, ident, W, seed, ckp in json.loads(Path(a.jobs).read_text()):
        name = tag(kind, N, ident, W, seed)
        if (OUT / 'E6B' / 'models' / f'{name}.npz').exists():
            print('SKIP', name); continue
        train(kind, N, ident, W, seed, steps=a.steps, checkpoints=ckp)
