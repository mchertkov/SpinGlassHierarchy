"""Stage 5E predeclared, bounded non-glassy N32/N64 controls.

Exactly sampled non-interacting conditional posteriors (no Markov-chain mixing
error), independently trained and evaluated fixed PCA trees, and the same
correlated gamma-grid dynamics as Stage 5B/5D. Seeds denote independent
training replicates, NOT quenched disorders of the non-glassy models.
"""
from pathlib import Path
import sys, json, argparse, time
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'code')]
from e3b_e1_heldout_hierarchy import train_tree,predict,adjusted_rand
from e5b_sequential_smc import GRID,entropy,latent_paths
OUT=ROOT/'results'/'E5E';OUT.mkdir(parents=True,exist_ok=True)

# The same two-state m0 as the earlier exact controls; calibrated *approximately*
# to a broad ±-pair overlap and never fitted to favorable Stage5E outcomes.
M0=.8
A=np.arctanh(M0)

def sample_prior(model,n,N,rng):
    if model=='product':return rng.choice(np.array([-1,1],np.int8),size=(n,N))
    if model=='two_state':
        z=rng.choice(np.array([-1,1],np.int8),size=(n,1))
        return (np.where(rng.random((n,N))<(1+M0)/2,1,-1)*z).astype(np.int8)
    raise ValueError(model)

def sample_posterior(model,H,draws,rng):
    """Exact iid samples conditional on each Gaussian observation H (npaths,N).

    Two-state sample latent ± then independent sites; product sites independent.
    """
    H=np.asarray(H,float);n,N=H.shape
    if model=='product':
        probs=.5*(1+np.tanh(H))
        return np.where(rng.random((n,draws,N))<probs[:,None,:],1,-1).astype(np.int8)
    if model=='two_state':
        lp=np.logaddexp(H+A,-H-A).sum(axis=-1)
        lm=np.logaddexp(H-A,-H+A).sum(axis=-1)
        p=np.exp(lp-np.logaddexp(lp,lm))
        lat=np.where(rng.random((n,draws))<p[:,None],1,-1)
        probs=.5*(1+np.tanh(H[:,None,:]+A*lat[:,:,None]))
        return np.where(rng.random((n,draws,N))<probs,1,-1).astype(np.int8)
    raise ValueError(model)

def overlap_relation(X,leaf):
    pi=np.bincount(leaf,minlength=4).astype(float)/len(leaf)
    means=np.vstack([X[leaf==j].mean(0) if np.any(leaf==j) else np.zeros(X.shape[1]) for j in range(4)])
    q=means@means.T/X.shape[1]
    result={}
    for label in ('same_leaf','siblings','separate_parent'):
        vals=[];ww=[]
        for i in range(4):
            for j in range(4):
                rel='same_leaf' if i==j else 'siblings' if i//2==j//2 else 'separate_parent'
                if rel==label:vals.append(q[i,j]);ww.append(pi[i]*pi[j])
        result[label]=float(np.average(vals,weights=ww)) if sum(ww)>0 else None
    return pi,result,q

def first_event(a,g):
    hit=np.flatnonzero(a>=.9)
    if len(hit)==0:return {'first90':None,'recross80':False}
    j=int(hit[0])
    return {'first90':float(g[j]),'recross80':bool(np.any(a[j+1:]<.8))}

def summarize(model,N,seed,paths=24,draws=2048):
    t=time.monotonic();r=np.random.default_rng(975317+N*10091+seed*1007+({'product':11,'two_state':91}[model]))
    trainA=sample_prior(model,12000,N,r)
    trainB=sample_prior(model,12000,N,r)
    treeA=train_tree(np.concatenate([trainA,-trainA]))
    treeB=train_tree(np.concatenate([trainB,-trainB]))
    hold=sample_prior(model,4000,N,r)
    clA,la=predict(hold,treeA);clB,lb=predict(hold,treeB)
    aris=[float(adjusted_rand(clA,clB)),float(adjusted_rand(la,lb))]
    ref=sample_prior(model,12000,N,r)
    prior_leaf=predict(ref,treeA)[1]
    pi,related,q=overlap_relation(ref,prior_leaf)
    gap=related['same_leaf']-related['siblings']
    accepted=bool(aris[0]>=.9 and aris[1]>=.7 and pi.min()>=.03 and gap>=.10)
    Y=sample_prior(model,paths,N,r)
    # Identical intrinsic PID schedule; same path law as planted Gaussian bridge.
    H=np.zeros((len(GRID),paths,N));last=0.
    for k,g in enumerate(GRID):
        dg=float(g-last)
        if dg>0:H[k]=H[k-1]+dg*Y+np.sqrt(dg)*r.standard_normal((paths,N))
        last=float(g)
    _,true_leaf=predict(Y,treeA)
    parent_prior=pi.reshape(2,2).sum(1)
    postmass=np.empty((paths,len(GRID),4),np.float32)
    mc_disagree=np.empty((paths,len(GRID)),float)
    for j,g in enumerate(GRID):
        z1=sample_posterior(model,H[j],draws,r)
        z2=sample_posterior(model,H[j],draws,r)
        a1=predict(z1.reshape(-1,N),treeA)[1].reshape(paths,draws)
        a2=predict(z2.reshape(-1,N),treeA)[1].reshape(paths,draws)
        mm1=np.stack([(a1==i).mean(axis=1) for i in range(4)],axis=1)
        mm2=np.stack([(a2==i).mean(axis=1) for i in range(4)],axis=1)
        postmass[:,j]=.5*(mm1+mm2)
        mc_disagree[:,j]=.5*np.abs(mm1-mm2).sum(axis=1)
    coarse=postmass.reshape(paths,len(GRID),2,2).sum(-1)
    Icoarse=entropy(parent_prior)-entropy(coarse).mean(0)
    Ifine=entropy(pi)-entropy(parent_prior)-entropy(postmass).mean(0)+entropy(coarse).mean(0)
    times=[]
    for I in (Icoarse,Ifine):
        ix=np.flatnonzero(I>=.5*I[-1]) if I[-1]>.01 else []
        times.append(float(GRID[ix[0]]) if len(ix) else None)
    events=[]
    for p in range(paths):
        leaf=int(true_leaf[p]);coarsep=coarse[p,:,leaf//2];finep=postmass[p,:,leaf]
        events.append(dict(path=p,coarse=first_event(coarsep,GRID),fine=first_event(finep,GRID)))
    out=dict(stage='5E',model=model,N=N,seed=seed,paths=paths,
        sample_counts=dict(training_each=12000,heldout=4000,prior_reference=12000,
                           posterior_draws_per_time_per_replicate=draws,independent_posterior_replicates=2),
        ARI_coarse=aris[0],ARI_fine=aris[1],prior_leaf_masses=pi.tolist(),
        overlap_by_relation=related,within_minus_sibling_overlap=gap,
        accepted_structural=accepted,information_half_times=times,
        terminal_information=[float(Icoarse[-1]),float(Ifine[-1])],
        coarse_commit_count=sum(e['coarse']['first90'] is not None for e in events),
        fine_commit_count=sum(e['fine']['first90'] is not None for e in events),
        coarse_recross_count=sum(e['coarse']['recross80'] for e in events),
        fine_recross_count=sum(e['fine']['recross80'] for e in events),
        mean_mc_population_tv=float(mc_disagree.mean()),max_mc_population_tv=float(mc_disagree.max()),
        gamma=GRID.tolist(),information_curves=np.stack([Icoarse,Ifine],axis=-1).tolist(),
        elapsed_seconds=float(time.monotonic()-t),
        note='Null model has no quenched disorder; seeds are independently sampled tree training replicates; posterior is sampled exactly conditional on observation up to iid Monte Carlo error, not by glass Gibbs dynamics. Times alone are not a Parisi-specific test.')
    name=f'{model}_N{N}_rep{seed}'
    (OUT/(name+'.json')).write_text(json.dumps(out,indent=2))
    np.savez_compressed(OUT/(name+'.npz'),gamma=GRID,Y=Y,H=H,leaf_mass=postmass,events=json.dumps(events),pi=pi,mc_tv=mc_disagree)
    print('E5E',model,N,seed,'ARI',np.round(aris,3),'overlap gap',round(gap,3),
          'structure',accepted,'times',times,'mcTV',round(float(mc_disagree.mean()),3),
          'seconds',round(out['elapsed_seconds'],2),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['product','two_state'],required=True)
    p.add_argument('--N',type=int,choices=[32,64],required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();summarize(a.model,a.N,a.seed)
