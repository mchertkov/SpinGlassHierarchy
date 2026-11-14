import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'experiments')]
import pidlib as P
from e5b_sequential_smc import systematic_resample,heatbath_rejuvenate,SequentialSMC,latent_paths
from e3b_e1_heldout_hierarchy import train_tree,predict


def test_systematic_resample_conditional_expected_offspring():
    w=np.array([.15,.25,.60]);r=np.random.default_rng(19)
    av=np.zeros(3)
    for _ in range(5000):av+=np.bincount(systematic_resample(w,r),minlength=3)
    assert np.max(np.abs(av/15000-w))<.005

def test_gaussian_observation_paths_fixed_endpoint():
    r=np.random.default_rng(10);y=np.array([1.,-1.,1.]);t=np.array([0,.1,.4])
    h=np.stack([latent_paths(y,t,r) for _ in range(3500)])
    assert np.max(abs(h[:,2].mean(0)-.4*y))<.045
    assert np.max(abs(h[:,2].var(0)-.4))<.04
    # Conditional Brownian-bridge point is the conditional Gaussian:
    a=np.array([.3,-.5,2.]);b=np.array([1.,0.,1.]);z=r.standard_normal((3500,3))
    hm=(a+b)/2+np.sqrt(.4/4)*z
    assert np.max(abs(hm.mean(0)-(a+b)/2))<.04
    assert np.max(abs(hm.var(0)-.1))<.025


def test_colored_gibbs_rejuvenation_preserves_exact_posterior():
    inst=P.bethe_spin_glass(8,3,1.4,'gauss',71);S=P.ExactSolver(inst)
    h=np.linspace(-.18,.22,8);r=np.random.default_rng(91)
    start=S.sample(h[None],12000,r)[0]
    end=heatbath_rejuvenate(start,inst,h,r,sweeps=8,global_flips=True)
    target=S.yhat(h[None])[0];m=end.mean(0)
    assert np.max(abs(m-target))<.07
    assert set(np.unique(end)).issubset({-1,1})


def test_particle_update_matches_exact_tiny_Ising_class_probability():
    inst=P.bethe_spin_glass(8,3,1.5,'gauss',12);S=P.ExactSolver(inst)
    r=np.random.default_rng(811);prior=S.sample_prior(1000,r)
    tr=train_tree(np.r_[prior,-prior]);Y=S.sample_prior(1,r)[0]
    grid=np.array([0.,.03,.1,.3,.8,1.3]);H=latent_paths(Y,grid,r)
    for rep in range(2):
        pop=SequentialSMC(S.sample_prior(1500,r),inst,tr,np.random.default_rng(900+rep),rejuvenation=12)
        for gamma,h in zip(grid[1:],H[1:]):pop.assimilate(float(gamma),h)
        _,labels=predict(S.X,tr)
        exact=np.bincount(labels,weights=S.weights(H[-1][None])[0],minlength=4)
        assert .5*abs(pop.mass()-exact).sum()<.13
        assert np.isclose(pop.weights.sum(),1)
        assert pop.resamples>=1
