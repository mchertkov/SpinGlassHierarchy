"""Tests of the identities used in the paper (tiny exactly enumerable systems).  Run: python3 tests/test_pidlib.py"""
import sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import pidlib as P

RNG = np.random.default_rng(0)


def small_instance(N=8, beta_J=1.2, seed=1):
    rng = np.random.default_rng(seed); e = P.random_regular_graph(N, 3, rng)
    return P.IsingInstance(N, e, rng.choice([-1.0, 1.0], len(e)), beta_J)


def test_bridge_limits():
    t = np.linspace(0.05, 0.95, 7); b0, b1 = P.Bridge(0.0), P.Bridge(1e-6)
    for f in ("a", "sigma2", "kappa", "c", "gamma"):
        assert np.allclose(getattr(b0, f)(t), getattr(b1, f)(t), rtol=1e-5), f
    b = P.Bridge(9.0); assert np.allclose(b.kappa(t), b.a(t) / b.sigma2(t)); assert np.allclose(b.t_of_gamma(b.gamma(t)), t, atol=1e-8)


def test_posterior_is_ising_in_moving_field():
    inst = small_instance(); S = P.ExactSolver(inst)
    for beta in (0.0, 4.0):
        b = P.Bridge(beta); t = 0.37; x = RNG.standard_normal((3, inst.N))
        a, s2 = b.a(t), b.sigma2(t)
        lp_bayes = S.logmu[None] - ((x[:, None, :] - a * S.X[None]) ** 2).sum(-1) / (2 * s2)   # Bayes with Gaussian bridge
        lp_bayes -= logsumexp(lp_bayes, 1, keepdims=True)
        lp_ising = S.logpost(b.kappa(t) * x); lp_ising -= logsumexp(lp_ising, 1, keepdims=True)
        assert np.allclose(lp_bayes, lp_ising, atol=1e-4)


def test_score_formula():
    inst = small_instance(N=6); S = P.ExactSolver(inst); b = P.Bridge(2.0); t = 0.4; x = RNG.standard_normal(inst.N)
    def logp(xx):
        a, s2 = b.a(t), b.sigma2(t)
        return logsumexp(S.logmu - ((xx - a * S.X) ** 2).sum(1) / (2 * s2))
    g = np.array([(logp(x + 1e-5 * e) - logp(x - 1e-5 * e)) / 2e-5 for e in np.eye(inst.N)])
    score = -x / b.sigma2(t) + b.kappa(t) * S.yhat(b.kappa(t) * x[None])[0]
    assert np.allclose(g, score, atol=1e-3)


def test_drift_is_posterior_average_of_bridge_drifts():
    inst = small_instance(); S = P.ExactSolver(inst); b = P.Bridge(3.0); t = 0.6; x = RNG.standard_normal((2, inst.N))
    w = S.weights(b.kappa(t) * x); ybar = w @ S.X
    u_avg = b.kappa(t) * ybar - b.c(t) * x
    u_formula = b.kappa(t) * S.yhat(b.kappa(t) * x) - b.c(t) * x
    assert np.allclose(u_avg, u_formula, atol=1e-5)


def test_orthogonal_increments_and_branching():
    inst = small_instance(N=10, beta_J=1.5); S = P.ExactSolver(inst); b = P.Bridge(0.0)
    Y = S.sample_prior(3000, RNG); tg = np.array([0.15, 0.4])
    r = P.order_parameter_paths(S, b, Y, tg, RNG)
    y1, y2 = r["Yh"][:, 0], r["Yh"][:, 1]
    lhs = np.mean((y2 * y1).sum(1)); rhs = np.mean((y1 * y1).sum(1))           # E yhat(t2).yhat(t1) = E|yhat(t1)|^2
    assert abs(lhs - rhs) < 0.05 * inst.N * 0.1 + 0.05 * abs(rhs)
    Qo, _, _ = P.branching_overlaps(S, b, r["X"][:, 0], tg[0], Y, 1, RNG)       # E Q(y, y') = E q(t_s)
    assert abs(Qo.mean() - r["q"][:, 0].mean()) < 0.03


def test_energy_information_identity():
    inst = small_instance(N=10, beta_J=1.0); S = P.ExactSolver(inst); b = P.Bridge(0.0)
    Y = S.sample_prior(4000, RNG); tg = np.r_[np.linspace(0.005, 0.1, 20), np.linspace(0.12, 0.6, 25)]
    r = P.energy_information_check(S, b, Y, tg, RNG)
    e, p = np.array(r["energy"]), np.array(r["prediction"])
    assert np.max(np.abs(e - p)[5:]) < 0.08 * np.max(np.abs(p)) + 0.05, (e[-1], p[-1])


def test_bp_exact_on_tree():
    N = 9; e = np.array([(i, (i - 1) // 2) for i in range(1, N)])[:, ::-1]; rng = np.random.default_rng(3)
    inst = P.IsingInstance(N, e, rng.standard_normal(len(e)), 1.3); H = rng.standard_normal((4, N))
    assert np.allclose(P.BPSolver(inst).yhat(H), P.ExactSolver(inst).yhat(H), atol=1e-5)


def test_pt_against_exact_energy():
    inst = small_instance(N=12, beta_J=1.3, seed=4); S = P.ExactSolver(inst)
    e_exact = (np.exp(S.logmu) * inst.energy(S.X)).sum()
    smp, _ = P.pt_sample(inst, np.linspace(0.3, 1.3, 8), 200, 800, 200, 10, np.random.default_rng(5))
    assert abs(inst.energy(smp.reshape(-1, inst.N)).mean() - e_exact) < 0.1, (inst.energy(smp.reshape(-1, inst.N)).mean(), e_exact)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)


def test_field_is_brownian_in_gamma():
    """h(t) = kappa(t) x(t) on a harmonic bridge to y: mean gamma*y, Cov(h_s,h_t) = gamma(min(s,t))."""
    b = P.Bridge(9.0); rng = np.random.default_rng(8); y = np.ones((20000, 1))
    tg = np.array([0.2, 0.5, 0.8]); X = b.bridge_paths(y, tg, rng, n_sub=200)[:, :, 0]
    H = X * b.kappa(tg)[None]; g = b.gamma(tg)
    assert np.allclose(H.mean(0), g, rtol=0.03, atol=0.03)
    C = np.cov(H.T); assert np.allclose(C, np.minimum.outer(g, g), rtol=0.06, atol=0.03), (C, np.minimum.outer(g, g))


if __name__ == "__main__":
    test_field_is_brownian_in_gamma(); print("ok test_field_is_brownian_in_gamma")
