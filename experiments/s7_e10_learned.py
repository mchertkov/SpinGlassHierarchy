"""E10 (step 3) -- Jacobian-spectrum route on trained H-PID denoisers (N=16 Bethe s1, exact reference).
Model: MLP  yhat_theta(h, gamma) ~ E[y | h],  h = gamma*y + sqrt(gamma)*z, trained by MSE (denoising objective).
Training sets: M in {64, 1024, inf (fresh exact samples each batch)}.
Measures: chi_theta = ||d yhat/dh||_F^2/N (autodiff), learned spectrum (Jacobian at small gamma),
reference-free calibration (Nishimori gap, rate gap), and the exact posterior on the same planted paths.
Usage: python3 experiments/e10_learned.py <M|inf> [steps]   -> results/E10_M<M>.json"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code"))
import pidlib_lite as P
torch.set_num_threads(1); torch.manual_seed(0)
Marg = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000; t0 = time.time()
inst = P.bethe_spin_glass(16, 3, 2.5, "gauss", 1); S = P.ExactSolver(inst); N = 16; mu = np.exp(S.logmu); mu /= mu.sum()
rng = np.random.default_rng(5)
train = None if Marg == "inf" else S.X[rng.choice(len(mu), int(Marg), p=mu)].astype(np.float32)

class Net(torch.nn.Module):
    def __init__(s, w=256):
        super().__init__(); s.f = torch.nn.Sequential(torch.nn.Linear(N + 1, w), torch.nn.SiLU(), torch.nn.Linear(w, w), torch.nn.SiLU(),
                                                   torch.nn.Linear(w, w), torch.nn.SiLU(), torch.nn.Linear(w, N))
    def forward(s, h, lg):          # input scaling: h / (gamma + sqrt(gamma)) ~ O(1);  log gamma as time feature
        g = torch.exp(lg)[:, None]; return torch.tanh(s.f(torch.cat([h / (g + g.sqrt()), lg[:, None] / 5], 1)))

net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
for it in range(steps):
    y = torch.tensor(S.X[rng.choice(len(mu), 512, p=mu)] if train is None else train[rng.integers(0, len(train), 512)])
    lg = torch.empty(512).uniform_(np.log(1e-3), np.log(10.0)); g = torch.exp(lg)[:, None]
    h = g * y + g.sqrt() * torch.randn_like(y)
    loss = ((net(h, lg) - y) ** 2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
t_train = time.time() - t0
# ---- evaluation on planted paths (truths from the exact measure, not the training set)
n = 300; gam = np.geomspace(1e-3, 10, 32); Y = S.X[rng.choice(len(mu), n, p=mu)].astype(float)
W = np.cumsum(rng.standard_normal((len(gam), n, N)) * np.sqrt(np.diff(np.r_[0, gam]))[:, None, None], 0)
iu = np.triu_indices(N, 1); Pp = (S.X[:, iu[0]] * S.X[:, iu[1]]).astype(np.float32)
from torch.func import jacrev, vmap
jac = vmap(jacrev(lambda hh, ll: net(hh[None], ll[None])[0], argnums=0))
out = {k: [] for k in ["q_th", "nish_gap", "chi_th", "chi_ex", "q_ex", "mse_th", "mse_ex", "lograte_th_std", "var_chi_th", "var_chi_ex"]}
for k, gg in enumerate(gam):
    H = gg * Y + W[k]; Ht = torch.tensor(H, dtype=torch.float32); lgt = torch.full((n,), float(np.log(gg)))
    with torch.no_grad(): yt = net(Ht, lgt).numpy()
    J = jac(Ht, lgt).detach().numpy(); chit = (J ** 2).sum((1, 2)) / N
    lp = S.logmu[None] + H.astype(np.float32) @ S.X.T; w = np.exp(lp - logsumexp(lp, 1, keepdims=True)).astype(np.float32)
    m = w @ S.X; off = w @ Pp - m[:, iu[0]] * m[:, iu[1]]; chie = (((1 - m ** 2) ** 2).sum(1) + 2 * (off ** 2).sum(1)) / N
    out["q_th"].append(float((yt ** 2).sum(1).mean() / N)); out["nish_gap"].append(float(((yt * Y).sum(1) - (yt ** 2).sum(1)).mean() / N))
    out["chi_th"].append(float(chit.mean())); out["chi_ex"].append(float(chie.mean())); out["q_ex"].append(float((m ** 2).sum(1).mean() / N))
    out["mse_th"].append(float(((yt - Y) ** 2).sum(1).mean())); out["mse_ex"].append(float(((m - Y) ** 2).sum(1).mean()))
    out["lograte_th_std"].append(float((gg * chit).std())); out["var_chi_th"].append(float(chit.var())); out["var_chi_ex"].append(float(chie.var()))
R = {k: np.array(v) for k, v in out.items()}
# learned spectrum: Jacobian at the smallest gamma, averaged over paths (estimates C_0 of the learned model)
H0 = torch.tensor(gam[0] * Y + W[0], dtype=torch.float32); J0 = jac(H0, torch.full((n,), float(np.log(gam[0])))).detach().numpy().mean(0)
lam_th = np.sort(np.linalg.eigvalsh(0.5 * (J0 + J0.T)))[::-1]
C0 = (S.X.T.astype(float) * mu) @ S.X.astype(float); lam_ex = np.sort(np.linalg.eigvalsh(C0))[::-1]
rate_gap = np.gradient(R["q_th"], gam) - R["chi_th"]
res = dict(M=Marg, steps=steps, train_seconds=t_train, gamma=gam.tolist(), **{k: v.tolist() for k, v in R.items()},
           lam_theta=lam_th[:6].tolist(), lam_exact=lam_ex[:6].tolist(), spectral_theta=(1 / lam_th[:3]).tolist(), spectral_exact=(1 / lam_ex[:3]).tolist(),
           rate_gap=rate_gap.tolist(), rel_chi_err_median=float(np.median(np.abs(R["chi_th"] - R["chi_ex"]) / R["chi_ex"])),
           peak_var_chi_theta=float(gam[np.argmax(R["var_chi_th"])]), peak_var_chi_exact=float(gam[np.argmax(R["var_chi_ex"])]),
           peak_lograte_theta=float(gam[np.argmax(gam * R["chi_th"])]), peak_lograte_exact=float(gam[np.argmax(gam * R["chi_ex"])]),
           max_abs_nish_gap=float(np.max(np.abs(R["nish_gap"]))), excess_mse_max=float(np.max(R["mse_th"] - R["mse_ex"])))
(ROOT / f"results/S7/E10_M{Marg}.json").write_text(json.dumps(res))
print(f"M={Marg} train {t_train:.0f}s | lam_theta={np.round(lam_th[:4], 2).tolist()} lam_exact={np.round(lam_ex[:4], 2).tolist()} | "
      f"chi rel err median={res['rel_chi_err_median']:.3f} | peaks lograte th/ex={res['peak_lograte_theta']:.3f}/{res['peak_lograte_exact']:.3f} "
      f"varchi th/ex={res['peak_var_chi_theta']:.3f}/{res['peak_var_chi_exact']:.3f} | max |Nishimori gap|={res['max_abs_nish_gap']:.3f} "
      f"| max excess MSE={res['excess_mse_max']:.3f} | rate gap max={np.max(np.abs(rate_gap)):.2f} | {time.time() - t0:.0f}s")
