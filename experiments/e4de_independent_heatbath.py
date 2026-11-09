"""Stage 4D-E: independent heat-bath PT, many planted observations, full time-series.

A genuinely independent update implementation from Metropolis/graph-color e4d_resumable.
For each disorder: 2F prior independent PT columns, F new planted observations,
6 posterior PT columns per observation (random vs ordered starting states).
No input configurations or measurements from earlier E4D posterior runs are reused.
The temperatures couple only the bond score; the Gaussian observation field is
held unchanged across all temperatures (correct target exp(beta*E + h.y)).

Each JSON/NPZ is a complete disorder-level checkpoint; any incomplete run is
not counted. Accept/reject is declared by downstream prespecified diagnostics.
"""
from pathlib import Path
import argparse,sys,json,time
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import pidlib as P

@njit(cache=True)
def run_pt(x, E, field, betas, nbr, Jn, sweeps, burn, stride, seed):
    """Sequential-site heatbath + nearest-neighbor PT; independent from Metropolis.

    Returns all retained cold configurations, bond energies, return counts, and
    swap acceptances. Random scan uses a new randomly rotated site order each sweep.
    """
    np.random.seed(seed)
    R,C,N=x.shape
    T=(sweeps-burn+stride-1)//stride
    cold=np.empty((T,C,N),np.int8)
    econ=np.empty((T,C),np.float64)
    tag=np.empty((R,C),np.int32)
    seen=np.zeros(C,np.bool_)
    trips=np.zeros(C,np.int32)
    acc=np.zeros(R-1,np.int64)
    att=np.zeros(R-1,np.int64)
    for r in range(R):
        for c in range(C): tag[r,c]=r
    saved=0
    for sw in range(sweeps):
        offset=np.random.randint(0,N)
        for r in range(R):
            for c in range(C):
                for ii in range(N):
                    i=(ii+offset)%N
                    loc=0.0
                    for k in range(nbr.shape[1]):
                        j=nbr[i,k]
                        if j<N: loc+=Jn[i,k]*x[r,c,j]
                    z=2*(betas[r]*loc+field[c,i])
                    if z>=0:
                        p=1/(1+np.exp(-z))
                    else:
                        ez=np.exp(z)
                        p=ez/(1+ez)
                    new=1 if np.random.random()<p else -1
                    old=x[r,c,i]
                    if new!=old:
                        x[r,c,i]=new
                        E[r,c]+=(new-old)*loc
        for r in range(sw%2,R-1,2):
            for c in range(C):
                att[r]+=1
                delta=(betas[r]-betas[r+1])*(E[r+1,c]-E[r,c])
                if np.log(max(1e-300,np.random.random()))<delta:
                    acc[r]+=1
                    for i in range(N):
                        tmp=x[r,c,i];x[r,c,i]=x[r+1,c,i];x[r+1,c,i]=tmp
                    en=E[r,c];E[r,c]=E[r+1,c];E[r+1,c]=en
                    tg=tag[r,c];tag[r,c]=tag[r+1,c];tag[r+1,c]=tg
        for c in range(C):
            pos=0
            for r in range(R):
                if tag[r,c]==0: pos=r;break
            if pos==R-1: seen[c]=True
            if pos==0 and seen[c]:trips[c]+=1;seen[c]=False
        if sw>=burn and (sw-burn)%stride==0:
            for c in range(C):
                econ[saved,c]=E[R-1,c]
                for i in range(N):cold[saved,c,i]=x[R-1,c,i]
            saved+=1
    return cold,econ,trips,acc,att

def initial(inst, betas, field, rng, half_ordered=True):
    R=len(betas);C=len(field);N=inst.N
    x=rng.choice(np.array([-1,1],np.int8),size=(R,C,N))
    if half_ordered: x[:,C//2:,:]=1
    E=inst.energy(x).astype(np.float64)
    return x,E

def produce(N,seed,fields=8,tau=3,beta_J=2.5,temps=28,prior_sweeps=12000,
            posterior_sweeps=18000,stride=20,output=None):
    tic=time.monotonic()
    out=Path(output) if output else ROOT/'results'/'E4DE_heatbath';out.mkdir(parents=True,exist_ok=True)
    base=f'HB_N{N}_seed{seed}_b{beta_J:g}_tau{tau:g}_fields{fields}_prior{prior_sweeps}_post{posterior_sweeps}'
    jf=out/(base+'.json');nf=out/(base+'.npz')
    if jf.exists() and nf.exists():
        print('CACHED',base,flush=True)
        return jf
    inst=P.bethe_spin_glass(N,3,beta_J,'gauss',seed)
    betas=beta_J*np.linspace(.000001,1.,temps)**1.15
    rng=np.random.default_rng(2026101700+N*10009+seed*73+int(tau*100)+int(beta_J*100))
    prior_h=np.zeros((2*fields,N))
    x,E=initial(inst,betas,prior_h,rng)
    if prior_sweeps<1200 or posterior_sweeps<1200: raise ValueError('insufficient minimum sweeps')
    prior,prior_e,prior_trip,prior_acc,prior_att=run_pt(x,E,prior_h,betas,inst.nbr,inst.Jn,
         prior_sweeps,prior_sweeps//2,max(20,stride),seed+110000+N)
    # Use F independent columns, half random and half ordered; fair sign flip
    # is a symmetry of the zero-field prior (not a replacement for equilibration).
    take=np.concatenate((np.arange(fields//2),fields+np.arange(fields-fields//2)))
    Y=prior[-1,take,:].copy()
    sign=rng.choice(np.array([-1,1],np.int8),size=(fields,1));Y*=sign
    gamma=tau/N
    h=gamma*Y+np.sqrt(gamma)*rng.standard_normal((fields,N))
    H=np.repeat(h,6,axis=0)
    x,E=initial(inst,betas,H,rng)
    post,post_e,post_trip,post_acc,post_att=run_pt(x,E,H,betas,inst.nbr,inst.Jn,
         posterior_sweeps,posterior_sweeps//2,stride,seed+220000+N)
    nrec=post.shape[0]
    # Each field has 3 independent random and 3 independent ordered chains.
    post=post.reshape(nrec,fields,2,3,N)
    post_e=post_e.reshape(nrec,fields,2,3)
    truth=(post*Y[None,:,None,None,:]).mean(axis=-1)
    Q01=np.mean(post[:,:,:,0]*post[:,:,:,1],axis=-1)
    Q02=np.mean(post[:,:,:,0]*post[:,:,:,2],axis=-1)
    Q12=np.mean(post[:,:,:,1]*post[:,:,:,2],axis=-1)
    q=np.stack([Q01,Q02,Q12],axis=-1)
    posth=np.einsum('tfacn,fn->tfac',post.astype(float),h)
    # independent original-chain time series available, not only last sample
    np.savez_compressed(nf,Y=Y,h=h,betas=betas,post=post,
        post_bond=post_e,post_field=posth,truth_q=truth,replica_q=q,
        prior_endpoints=prior[:,take,:],prior_bond=prior_e[:,take],
        prior_roundtrips=prior_trip,posterior_roundtrips=post_trip.reshape(fields,2,3),
        posterior_swaps=post_acc/np.maximum(post_att,1),
        prior_swaps=prior_acc/np.maximum(prior_att,1))
    # Full energy audit for last saved spin; valid for sampled configurations too.
    err=np.max(np.abs(inst.energy(post[-1].reshape(-1,N))-post_e[-1].reshape(-1)))
    assert err<1e-5,('bond-energy drift',err)
    info={'stage':'4D-E','algorithm':'sequential-site heat-bath + independent parallel tempering',
      'N':N,'seed':seed,'beta_J':beta_J,'tau':tau,'gamma':gamma,'fields':fields,
      'prior_sweeps':prior_sweeps,'posterior_sweeps':posterior_sweeps,'stride':stride,
      'posterior_saved_samples':nrec,'prior_min_roundtrips':int(prior_trip.min()),
      'posterior_min_roundtrips':int(post_trip.min()),
      'posterior_roundtrips_per_field':post_trip.reshape(fields,2,3).tolist(),
      'prior_roundtrips_per_chain':prior_trip.tolist(),
      'posterior_swap_acceptance_min':float(np.min(post_acc/np.maximum(post_att,1))),
      'prior_swap_acceptance_min':float(np.min(prior_acc/np.maximum(prior_att,1))),
      'max_bond_energy_error':float(err),'wall_seconds':float(time.monotonic()-tic),
      'note':'Both starts are random versus ordered. Holding observation h fixed at every ladder temperature is intentional. Passing operational tests is not proof of rare-state equilibration.'}
    jf.write_text(json.dumps(info,indent=2))
    print('COMPLETE',base,'time',round(info['wall_seconds'],1),'post_min_rt',info['posterior_min_roundtrips'],
        'prior_min_rt',info['prior_min_roundtrips'],flush=True)
    return jf

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('N',type=int);a.add_argument('seed',type=int)
    a.add_argument('--fields',type=int,default=8);a.add_argument('--tau',type=float,default=3)
    a.add_argument('--beta-J',type=float,default=2.5);a.add_argument('--prior-sweeps',type=int,default=12000)
    a.add_argument('--posterior-sweeps',type=int,default=18000);a.add_argument('--temps',type=int,default=28)
    a.add_argument('--stride',type=int,default=20);a.add_argument('--output')
    args=a.parse_args()
    produce(args.N,args.seed,args.fields,args.tau,args.beta_J,args.temps,args.prior_sweeps,args.posterior_sweeps,args.stride,args.output)
