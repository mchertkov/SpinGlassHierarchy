"""Validation experiments used in Secs. 7.2, 7.3 and 7.5.
  python experiments/s8_validation.py e0                         blind identification on the E0 model (N=500)
  python experiments/s8_validation.py structured                 nested non-glassy control at N=16 (exact, same pipeline as the glasses)
  python experiments/s8_validation.py tscan <beta> <s0> <s1>     unselected Bethe instances at N=16, seeds s0..s1-1, coupling scale beta
Outputs in results/S8/.  The identification, acceptance tests and gamma grid are those of experiments/e5_exact_transients.py."""
from pathlib import Path
import sys, json, time
import numpy as np
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / "code"), str(ROOT / "experiments")]
import pidlib as P
from e3b_e1_heldout_hierarchy import train_tree, predict, adjusted_rand
OUT = ROOT / "results" / "S8"; OUT.mkdir(parents=True, exist_ok=True)
GAMMA = np.array([0, .005, .012, .025, .045, .07, .105, .15, .21, .28, .38, .5, .65, .85, 1.1, 1.45, 1.9, 2.5, 3.5, 5.])


def entropy(a):
    return -np.sum(a * np.log(np.maximum(a, 1e-300)), axis=-1)


def half_time(g, frac, lam=0.5):
    for k in range(1, len(g)):
        if frac[k - 1] < lam <= frac[k]:
            if g[k - 1] <= 0: return float(g[k] * lam / max(frac[k], 1e-12))
            f = (lam - frac[k - 1]) / (frac[k] - frac[k - 1]); return float(np.exp(np.log(g[k - 1]) + f * (np.log(g[k]) - np.log(g[k - 1]))))
    return float("nan")


def identify(X, mu, sample, rng):
    """One independent run of the identification protocol (two fits + held-out ARI, masses, overlap gap)."""
    N = X.shape[1]; t1, t2 = sample(12000, rng), sample(12000, rng)
    tA, tB = train_tree(np.r_[t1, -t1]), train_tree(np.r_[t2, -t2]); held = sample(5000, rng)
    ca, fa = predict(held, tA); cb, fb = predict(held, tB); aris = [adjusted_rand(ca, cb), adjusted_rand(fa, fb)]
    cX, fX = predict(X, tA); M = (fX[:, None] == np.arange(4)[None, :]).astype(float); mass = mu @ M
    means = np.stack([(mu * M[:, l]) @ X / max(mass[l], 1e-300) for l in range(4)]); Q = means @ means.T / N
    rel = {"same_leaf": [], "siblings": []}
    for i in range(4):
        for j in range(4):
            if i == j: rel["same_leaf"].append((Q[i, j], mass[i] * mass[j]))
            elif i // 2 == j // 2: rel["siblings"].append((Q[i, j], mass[i] * mass[j]))
    gap = float(np.average([v for v, _ in rel["same_leaf"]], weights=[w for _, w in rel["same_leaf"]]) -
                np.average([v for v, _ in rel["siblings"]], weights=[w for _, w in rel["siblings"]]))
    accepted = bool(aris[0] >= .9 and aris[1] >= .70 and mass.min() >= .03 and gap >= .10)
    return dict(aris=aris, masses=mass.tolist(), gap=gap, accepted=accepted), M, mass


def assay(X, logmu, sample, rng, npaths=48, runs=3):
    """Exact N<=16 assay: three independent identification runs + level information along planted paths (tree of run 1)."""
    mu = np.exp(logmu - logsumexp(logmu)); N = X.shape[1]
    ids = [identify(X, mu, sample, rng) for _ in range(runs)]
    first, M, mass = ids[0]
    C0 = (X.T * mu) @ X; lam = np.sort(np.linalg.eigvalsh(C0))[::-1]
    Y = sample(npaths, rng); H = np.zeros((len(GAMMA), npaths, N)); h = np.zeros((npaths, N)); last = 0.
    for k, g in enumerate(GAMMA):
        if g > last: h += (g - last) * Y + np.sqrt(g - last) * rng.standard_normal((npaths, N))
        H[k] = h; last = g
    pp = mass.reshape(2, 2).sum(-1); info = np.zeros((len(GAMMA), 2))
    for k in range(len(GAMMA)):
        lw = H[k] @ X.T + logmu[None]; w = np.exp(lw - logsumexp(lw, 1, keepdims=True)); leaf = w @ M; par = leaf.reshape(-1, 2, 2).sum(-1)
        info[k] = [np.mean(entropy(pp) - entropy(par)), np.mean(entropy(mass) - entropy(pp) - (entropy(leaf) - entropy(par)))]
    hc = half_time(GAMMA, info[:, 0] / max(info[-1, 0], 1e-12)); hf = half_time(GAMMA, info[:, 1] / max(info[-1, 1], 1e-12))
    acc = [r[0]["accepted"] for r in ids]
    return dict(**first, acceptance_runs=acc, acceptance_rate=float(np.mean(acc)), robust=bool(all(acc)),
                half=[hc, hf], terminal_info=info[-1].tolist(), lam=lam[:4].tolist())


def tscan(beta, s0, s1):
    out = OUT / "tscan"; out.mkdir(exist_ok=True); t0 = time.time()
    for seed in range(s0, s1):
        f = out / f"beta{beta}_seed{seed}.json"
        if f.exists(): continue
        inst = P.bethe_spin_glass(16, 3, beta, "gauss", seed); S = P.ExactSolver(inst); X = S.X.astype(float)
        rng = np.random.default_rng(900000 + seed); mu = np.exp(S.logmu); mu /= mu.sum()
        sample = lambda n, r: X[r.choice(len(X), n, p=mu)]
        r = assay(X, S.logmu.astype(float), sample, rng); r.update(beta=beta, seed=seed); f.write_text(json.dumps(r))
    print(f"beta={beta} seeds {s0}-{s1} done in {time.time() - t0:.0f}s", flush=True)


def structured(n_rep=5, m0=0.8, flip=0.25, N=16):
    res = []; idx = np.arange(2 ** N)
    X = (1 - 2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1)).astype(float)
    for rep in range(n_rep):
        rg = np.random.default_rng(5000 + rep); u = np.where(rg.random(N) < .5, 1., -1.)
        kids = [np.where(rg.random(N) < flip, -u, u) for _ in range(2)]
        m = m0 * np.array(kids + [-k for k in kids]); W = np.full(4, .25)
        lp = np.stack([np.log((1 + X * mm) / 2).sum(1) for mm in m], 1) + np.log(W); logmu = logsumexp(lp, 1)
        def sample(n, r, m=m):
            a = r.integers(0, 4, n); return np.where(r.random((n, N)) < (1 + m[a]) / 2, 1., -1.)
        r = assay(X, logmu, sample, np.random.default_rng(6000 + rep)); r.update(rep=rep, m0=m0, flip=flip); res.append(r)
        print("structured rep", rep, "ARI", np.round(r["aris"], 2), "gap", round(r["gap"], 2), "accepted", r["accepted"], "half", np.round(r["half"], 3), flush=True)
    (OUT / "structured_null.json").write_text(json.dumps(res, indent=1))


def e0(npaths=300, K=96):
    """Blind identification on E0: trees trained on unlabeled samples; recovered vs true classes; decision times on both."""
    S = P.ultrametric_caricature(N=500); N = S.N; lab = S.labels; rng = np.random.default_rng(31)
    def sample_lab(n):
        a = rng.choice(S.A, n, p=S.W); return np.where(rng.random((n, N)) < (1 + S.m[a]) / 2, 1., -1.), a
    t1, _ = sample_lab(12000); t2, _ = sample_lab(12000)
    tA, tB = train_tree(np.r_[t1, -t1]), train_tree(np.r_[t2, -t2]); held, ah = sample_lab(5000)
    ca, fa = predict(held, tA); cb, fb = predict(held, tB)
    true_c = lab[ah, 0]; true_f = 2 * lab[ah, 0] + lab[ah, 1]
    out = dict(ARI_between_fits=[adjusted_rand(ca, cb), adjusted_rand(fa, fb)], ARI_vs_truth=[adjusted_rand(ca, true_c), adjusted_rand(fa, true_f)])
    # decision times: true classes (exact state posterior) vs recovered classes (posterior continuations classified by the tree)
    Y, aY = sample_lab(npaths); gam = np.geomspace(1e-4, 5, 30); h = np.zeros((npaths, N)); last = 0.
    yc_rec, yf_rec = predict(Y, tA); yc_true, yf_true = lab[aY, 0], 2 * lab[aY, 0] + lab[aY, 1]
    prior_c = np.array([S.W[lab[:, 0] == c].sum() for c in (0, 1)]); prior_f = np.array([S.W[(2 * lab[:, 0] + lab[:, 1]) == f].sum() for f in range(4)])
    Ct, Ft, Cr, Fr = [], [], [], []
    for g in gam:
        h += (g - last) * Y + np.sqrt(g - last) * rng.standard_normal((npaths, N)); last = g
        post = S.post_states(h)
        pc = np.stack([post[:, lab[:, 0] == c].sum(1) for c in (0, 1)], 1); pf = np.stack([post[:, (2 * lab[:, 0] + lab[:, 1]) == f].sum(1) for f in range(4)], 1)
        Ct.append(np.mean(np.log(pc[np.arange(npaths), yc_true] / prior_c[yc_true])))
        Ft.append(np.mean(np.log(pf[np.arange(npaths), yf_true] / prior_f[yf_true])) - Ct[-1])
        conts = S.sample(h, K, rng); cc, ff = predict(conts.reshape(-1, N), tA); cc = cc.reshape(npaths, K); ff = ff.reshape(npaths, K)
        pcr = (np.stack([(cc == c).sum(1) for c in (0, 1)], 1) + .5) / (K + 1); pfr = (np.stack([(ff == f).sum(1) for f in range(4)], 1) + .5) / (K + 1)
        Cr.append(np.mean(np.log(pcr[np.arange(npaths), yc_rec] / .5))); Fr.append(np.mean(np.log(pfr[np.arange(npaths), yf_rec] / .25)) - Cr[-1])
    Ct, Ft, Cr, Fr = map(np.array, (Ct, Ft, Cr, Fr))
    out.update(half_true=[half_time(gam, Ct / Ct[-1]), half_time(gam, Ft / Ft[-1])], half_recovered=[half_time(gam, Cr / Cr[-1]), half_time(gam, Fr / Fr[-1])],
               gamma=gam.tolist(), info_true=[Ct.tolist(), Ft.tolist()], info_recovered=[Cr.tolist(), Fr.tolist()])
    (OUT / "e0_blind.json").write_text(json.dumps(out, indent=1))
    print("E0 blind:", {k: (np.round(v, 3).tolist() if isinstance(v, list) and len(v) == 2 else "...") for k, v in out.items()})


if __name__ == "__main__" and sys.argv[1] not in ("structured_truth", "summary"):
    a = sys.argv
    if a[1] == "e0": e0()
    elif a[1] == "structured": structured()
    else: tscan(float(a[2]), int(a[3]), int(a[4]))


def structured_truth(n_rep=5, m0=0.8, flip=0.25, N=16):
    """Blind recovery for the structured control: ARI of the identified tree against the latent truth (same seeds as structured())."""
    out = []
    for rep in range(n_rep):
        rg = np.random.default_rng(5000 + rep); u = np.where(rg.random(N) < .5, 1., -1.)
        kids = [np.where(rg.random(N) < flip, -u, u) for _ in range(2)]; m = m0 * np.array(kids + [-k for k in kids])
        r = np.random.default_rng(7000 + rep)
        def samp(n):
            a = r.integers(0, 4, n); return np.where(r.random((n, N)) < (1 + m[a]) / 2, 1., -1.), a
        t, _ = samp(12000); tr = train_tree(np.r_[t, -t]); held, a = samp(5000); c, f = predict(held, tr)
        out.append(dict(rep=rep, ARI_coarse_vs_truth=adjusted_rand(c, (a >= 2).astype(int)), ARI_fine_vs_truth=adjusted_rand(f, a)))
    (OUT / "structured_truth.json").write_text(json.dumps(out, indent=1)); print(out)


def summary():
    import glob
    R = [json.load(open(f)) for f in glob.glob(str(OUT / "tscan" / "*.json"))]; res = {}
    for b in sorted({r["beta"] for r in R}):
        G = [r for r in R if r["beta"] == b]; rob = [r for r in G if r["robust"]]; anyacc = [r for r in G if r["acceptance_rate"] > 0]
        ratio = [r["half"][1] / r["half"][0] for r in rob if np.isfinite(r["half"][0]) and np.isfinite(r["half"][1]) and r["half"][0] > 0]
        res[str(b)] = dict(n=len(G), robust_fraction=len(rob) / len(G), any_run_fraction=len(anyacc) / len(G),
                           mean_acceptance_rate=float(np.mean([r["acceptance_rate"] for r in G])),
                           median_coarse_half_robust=float(np.median([r["half"][0] for r in rob])) if rob else None,
                           median_fine_half_robust=float(np.median([r["half"][1] for r in rob])) if rob else None,
                           median_fine_over_coarse=float(np.median(ratio)) if ratio else None,
                           fraction_fine_later=float(np.mean([x > 1 for x in ratio])) if ratio else None,
                           median_lambda1_over_N=float(np.median([r["lam"][0] / 16 for r in G])),
                           median_coarse_ARI=float(np.median([r["aris"][0] for r in G])), median_fine_ARI=float(np.median([r["aris"][1] for r in G])))
    (OUT / "tscan_summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__" and sys.argv[1] in ("structured_truth", "summary"):
    {"structured_truth": structured_truth, "summary": summary}[sys.argv[1]]()
