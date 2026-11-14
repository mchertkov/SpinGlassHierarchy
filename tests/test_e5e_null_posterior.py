"""New Stage5E direct-sampling and structural protocol regression tests."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'code')]
from e5e_matched_nulls import sample_prior,sample_posterior,overlap_relation,first_event


def _exact(model,h,m0=.8):
    n=len(h);ints=np.arange(2**n,dtype=np.int32)[:,None]
    X=np.where(ints&(1<<np.arange(n)),1.,-1.)
    if model=='product': prior=np.ones(len(X))
    else:prior=.5*np.exp(np.arctanh(m0)*X.sum(1))+.5*np.exp(-np.arctanh(m0)*X.sum(1))
    w=prior*np.exp(X@h-np.max(X@h));w/=w.sum()
    return w@X


def test_sample_posterior_matches_exact_small_N():
    r=np.random.default_rng(1943)
    H=np.array([[0.,.13,-.7,.3,.8,-.2,.01],[-1.,0.,.01,.4,.2,.3,1.]])
    for model in ('product','two_state'):
        draws=sample_posterior(model,H,40000,r)
        assert draws.shape==(2,40000,7)
        assert set(np.unique(draws))=={-1,1}
        for j in range(len(H)):
            target=_exact(model,H[j])
            np.testing.assert_allclose(draws[j].mean(0),target,atol=.025)


def test_prior_two_state_signed_symmetry():
    r=np.random.default_rng(210)
    for model in ('product','two_state'):
        x=sample_prior(model,15000,32,r)
        assert abs(x.mean())<.03  # common latent sign induces O(n_samples^-1/2) fluctuations


def test_overlap_gap_null_properties():
    X=np.array([[1,1],[1,-1],[-1,1],[-1,-1]],dtype=float)
    pi,rel,Q=overlap_relation(X,np.arange(4))
    assert np.allclose(pi,.25)
    assert rel['same_leaf']==1
    assert abs(rel['siblings'])<1e-12


def test_event_grid_nonabsorbing():
    g=np.array([0.,.1,.2,.3])
    d=first_event(np.array([.4,.92,.79,.94]),g)
    assert d=={'first90':.1,'recross80':True}
