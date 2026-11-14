"""Stage 6 regression tests (fast)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'code')]


def test_n16_tree_replay_matches_frozen():
    from e6_common import ExactHierarchy
    E = ExactHierarchy('bethe', 4)
    d = json.loads((ROOT / 'results' / 'E5F' / 'exact16' / 'bethe_seed4_p80.json').read_text())
    assert np.allclose(E.pi, d['prior_leaf_masses'], atol=1e-9)
    assert np.allclose(E.aris, d['heldout_ARI'], atol=1e-12)


def test_budget_and_contrast_identities_exact():
    from e6_common import ExactHierarchy, level_quantities
    E = ExactHierarchy('bethe', 3)
    rng = np.random.default_rng(0)
    H = 0.3 * rng.standard_normal((5, 16))
    leaf, mom, m = E.moments(H)
    _, _, rc, rf = level_quantities(leaf, mom, m)
    w = E.posterior(H)
    for b in range(5):
        C = (E.X.T * w[b]) @ E.X - np.outer(w[b] @ E.X, w[b] @ E.X)
        within = .5 * (16 - (np.sum(mom[b] ** 2, -1) / leaf[b]).sum())
        assert abs(.5 * np.trace(C) - (rc[b] + rf[b] + within)) < 1e-4


def test_jacobian_matches_autograd():
    import pytest
    torch = pytest.importorskip("torch")   # optional dependency, only needed for the learned denoisers
    from e6b_train import Denoiser
    from e6b_analyze import jac_sym
    torch.manual_seed(0); net = Denoiser(8, 4)
    H = np.random.default_rng(1).standard_normal((3, 8))
    J = jac_sym(net, H, .3)
    for b in range(3):
        f = lambda h: net(h[None], torch.tensor([np.log(.3)], dtype=torch.float32))[0]
        Ja = torch.autograd.functional.jacobian(f, torch.tensor(H[b], dtype=torch.float32)).numpy()
        assert np.allclose(J[b], .5 * (Ja + Ja.T), atol=1e-5)
