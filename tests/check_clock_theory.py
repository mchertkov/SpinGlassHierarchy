"""Fast deterministic checks of the Stage-1 corrections (no frozen E0/E1 data reused).
Run: python3 tests/test_stage1_theory.py. Uses analytic Ising-product/latent binary models.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.special import roots_hermitenorm
from scipy.integrate import quad
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from pidlib import Bridge

z, w = roots_hermitenorm(90)
w /= np.sqrt(2*np.pi)

def expectation(f):
    return float(np.sum(w*f(z)))

def assert_close(name, got, expected, tolerance):
    err=abs(got-expected)
    assert err<tolerance, (name, got, expected, err)
    print(f'PASS {name}: absolute residual={err:.3e}')

def class_info(gamma, rho):
    h = gamma + np.sqrt(gamma)*z     # symmetry reduces to Y=+1
    m = np.tanh(h)
    return float(np.sum(w * ((1+rho)/2 * np.log1p(rho*m) + (1-rho)/2 * np.log1p(-rho*m))))

def class_rate(gamma, rho):
    h = gamma+np.sqrt(gamma)*z
    m=np.tanh(h)
    pc=(1+rho*m)/2
    mplus=np.tanh(h+np.arctanh(rho))
    mminus=np.tanh(h-np.arctanh(rho))
    return .5 * expectation(lambda zz: pc*(mplus-m)**2+(1-pc)*(mminus-m)**2)

def I_rad(gamma):
    h=gamma+np.sqrt(gamma)*z
    return gamma - expectation(lambda zz: np.logaddexp(h,-h)-np.log(2.))

def kinetic_rad(b,t):
    a=float(b.a(t)); sd=np.sqrt(float(b.sigma2(t)))
    x=a+sd*z
    h=b.kappa(t)*x
    return expectation(lambda zz: (b.kappa(t)*np.tanh(h)-b.c(t)*x)**2)

def a_prime(b,t):
    S=b.s
    return 1.0 if S==0 else float(S*np.cosh(S*t)/np.sinh(S))

def main():
    for beta in (0., 4., 36.):
        b=Bridge(beta)
        s,t=.19,.72
        if beta==0:
            A=(1-t)/(1-s); B=(t-s)/(1-s); V=(t-s)*(1-t)/(1-s)
        else:
            S=np.sqrt(beta)
            A=np.sinh(S*(1-t))/np.sinh(S*(1-s))
            B=np.sinh(S*(t-s))/np.sinh(S*(1-s))
            V=np.sinh(S*(t-s))*np.sinh(S*(1-t))/(S*np.sinh(S*(1-s)))
        assert_close(f'bridge mean beta={beta}', A*b.a(s)+B,b.a(t),1e-12)
        assert_close(f'bridge variance beta={beta}',A*A*b.sigma2(s)+V,b.sigma2(t),1e-12)
        for tm in (.1,.4,.7):
            dg=(b.gamma(tm+1e-6)-b.gamma(tm-1e-6))/(2e-6)
            assert_close(f'gamma derivative beta={beta} t={tm}',dg,b.kappa(tm)**2,3e-7)
        tE=.58
        energy= .5*quad(lambda tt: kinetic_rad(b,tt),0,tE,epsabs=1e-8,limit=120)[0]
        baseline=.5*quad(lambda tt: a_prime(b,tt)**2+b.c(tt)**2*b.sigma2(tt),0,tE,epsabs=1e-10)[0]
        assert_close(f'kinetic information beta={beta}',energy,baseline-I_rad(float(b.gamma(tE))),3e-7)
    for rho in (.25,.65,.9):
        for g in (.15,.6,2.):
            eps=2e-5
            deriv=(class_info(g+eps,rho)-class_info(g-eps,rho))/(2*eps)
            assert_close(f'latent class information rate rho={rho} gamma={g}',deriv,class_rate(g,rho),5e-7)
    for N,t0,tau in ((1000,.4,1.),(100000,.55,1.),(1000000,.75,2.)):
        f=lambda S: N*Bridge(S*S).gamma(t0)-tau
        S=brentq(f,0.1,70.)
        assert_close(f'fixed physical time N={N}',N*Bridge(S*S).gamma(t0),tau,5e-10)
        print(f' stiffness root N={N}: sqrt(beta)={S:.6f}, beta={S*S:.6f}')
    print('All fast Stage-1 theory checks passed.')

if __name__=='__main__': main()
