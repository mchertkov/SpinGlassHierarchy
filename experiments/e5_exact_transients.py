"""Stage 5: exact N=16, correlated-field, predeclared finite-size hierarchy assay.
Authoritative source: e4dh_dynamic_tree_exact.py; extensions add nulls, per-path crossings,
bootstrap clock intervals, and prior/leaf overlap checks. No manuscript changes.
"""
from pathlib import Path
import sys, json, argparse, time
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'experiments')]
import pidlib as P
from e3b_e1_heldout_hierarchy import train_tree, predict, adjusted_rand
OUT=ROOT/'results'/'E5F'/'exact16'; OUT.mkdir(parents=True,exist_ok=True)
GAMMA=np.array([0,.005,.012,.025,.045,.07,.105,.15,.21,.28,.38,.5,.65,.85,1.1,1.45,1.9,2.5,3.5,5.])

def setup(model, seed):
    inst=P.bethe_spin_glass(16,3,2.5,'gauss',seed)
    S=P.ExactSolver(inst)
    if model=='bethe':return S
    if model=='product':
        S.logmu[:]=-np.log(len(S.X))
        def dr(n,r):return r.choice([-1,1],size=(n,16)).astype(float)
    elif model=='two_state':
        m0=.8;X=S.X.astype(float)
        lp=(np.log((1+m0)/2)*(X==1)+np.log((1-m0)/2)*(X==-1)).sum(-1)
        lm=(np.log((1-m0)/2)*(X==1)+np.log((1+m0)/2)*(X==-1)).sum(-1)
        S.logmu=np.logaddexp(lp,lm)-np.log(2)
        def dr(n,r):
            latent=r.choice([-1,1],size=(n,1));return (np.where(r.random((n,16))<(1+m0)/2,1,-1)*latent).astype(float)
    else:raise ValueError(model)
    S.sample_prior=dr
    return S

def entropy(a):return -np.sum(a*np.log(np.maximum(a,1e-300)),axis=-1)

def analyze(model,seed,npaths=96):
    tic=time.monotonic()
    rng=np.random.default_rng(202611190+seed*101+{'bethe':0,'two_state':1,'product':2}[model]*1000000)
    S=setup(model,seed);X=S.X.astype(float);mu=np.exp(S.logmu);mu/=mu.sum()
    t1=S.sample_prior(12000,rng);t2=S.sample_prior(12000,rng)
    treeA=train_tree(np.r_[t1,-t1]);treeB=train_tree(np.r_[t2,-t2])
    held=S.sample_prior(5000,rng)
    ca,fa=predict(held,treeA);cb,fb=predict(held,treeB)
    aris=[adjusted_rand(ca,cb),adjusted_rand(fa,fb)]
    cX,fX=predict(X,treeA)
    membership=(fX[:,None]==np.arange(4)[None,:]).astype(float)
    mass_prior=mu@membership;parent_prior=mass_prior.reshape(2,2).sum(-1)
    means=np.empty((4,16))
    for l in range(4):means[l]=(mu*membership[:,l])@X/max(mass_prior[l],1e-300)
    Q=means@means.T/16
    weighted={}
    for typ in ('same_leaf','siblings','separate_parent'):
        vals=[];ww=[]
        for i in range(4):
            for j in range(4):
                relation=('same_leaf' if i==j else 'siblings' if i//2==j//2 else 'separate_parent')
                if typ==relation:
                    vals.append(Q[i,j]);ww.append(mass_prior[i]*mass_prior[j])
        weighted[typ]=float(np.average(vals,weights=ww))
    Y=S.sample_prior(npaths,rng); ycoarse,yleaf=predict(Y,treeA)
    H=np.empty((len(GAMMA),npaths,16));h=np.zeros((npaths,16));last=0.
    for k,g in enumerate(GAMMA):
        dg=g-last
        if dg>0:h+=dg*Y+np.sqrt(dg)*rng.standard_normal((npaths,16))
        H[k]=h;last=g
    pm=np.empty((len(GAMMA),npaths,4),np.float32)
    info=np.empty((len(GAMMA),2));rates=np.empty_like(info)
    per_path_info=np.empty((len(GAMMA),npaths,2))
    for k,g in enumerate(GAMMA):
        coarse_arr=[];fine_arr=[];rr0=[];rr1=[]
        for begin in range(0,npaths,24):
            sl=slice(begin,min(npaths,begin+24))
            lw=H[k,sl]@X.T+S.logmu[None,:]
            w=np.exp(lw-logsumexp(lw,axis=1,keepdims=True))
            leaf=w@membership
            pm[k,sl]=leaf
            parent=leaf.reshape(-1,2,2).sum(-1)
            coarse_arr.append(entropy(parent));fine_arr.append(entropy(leaf)-entropy(parent))
            # Level information-rate estimators from moment decompositions
            m=w@X
            leaf_m=np.stack([(w*membership[:,j])@X for j in range(4)],axis=1)
            parent_m=leaf_m.reshape(-1,2,2,16).sum(axis=2)
            E1=(np.sum(parent_m**2,-1)/np.maximum(parent,1e-50)).sum(-1)
            E2=(np.sum(leaf_m**2,-1)/np.maximum(leaf,1e-50)).sum(-1)
            rr0.append(.5*(E1-np.sum(m*m,axis=-1)))
            rr1.append(.5*(E2-E1))
            per_path_info[k,sl,0]=entropy(parent_prior)-entropy(parent)
            per_path_info[k,sl,1]=entropy(mass_prior)-entropy(parent_prior)-(entropy(leaf)-entropy(parent))
        info[k]=per_path_info[k].mean(0)
        rates[k]=[np.concatenate(rr0).mean(),np.concatenate(rr1).mean()]
    P=pm.reshape(len(GAMMA),npaths,2,2).sum(-1)
    parenttrue=P[np.arange(len(GAMMA))[:,None],np.arange(npaths)[None,:],ycoarse[None,:]]
    leaftrue=pm[np.arange(len(GAMMA))[:,None],np.arange(npaths)[None,:],yleaf[None,:]]
    def crossings(arr):
        hit=(arr>=.9);ever=hit.any(axis=0);first=hit.argmax(axis=0)
        recross=np.array([bool(np.any(arr[first[i]+1:,i]<.8)) if ever[i] else False for i in range(npaths)])
        return dict(hitting_fraction=float(ever.mean()),recross_fraction_all=float(recross.mean()),
                    recross_fraction_among_hit=float(recross.sum()/max(ever.sum(),1)),
                    median_hitting_gamma=float(np.median(GAMMA[first[ever]])) if ever.any() else None)
    def clock_curve(x):
        out=[]
        for lev in range(2):
            ix=np.flatnonzero(x[:,lev]>=.5*x[-1,lev])
            out.append(float(GAMMA[ix[0]]) if len(ix) and x[-1,lev]>0 else None)
        return out
    clock=clock_curve(info)
    b_rng=np.random.default_rng(811902+seed*100+{'bethe':0,'two_state':1,'product':2}[model])
    bclock=[]
    for ib in range(300):
        ii=b_rng.integers(0,npaths,npaths)
        bclock.append(clock_curve(per_path_info[:,ii].mean(1)))
    bclock=np.array([[np.nan if v is None else v for v in pair] for pair in bclock])
    ci=np.nanpercentile(bclock,[2.5,97.5],axis=0).T
    result=dict(kind=model,seed=seed,N=16,paths=npaths,method='exact enumeration 2**16; planted correlated Gaussian paths',
          gamma=GAMMA.tolist(),heldout_ARI=aris,prior_leaf_masses=mass_prior.tolist(),prior_class_overlap_matrix=Q.tolist(),
          prior_overlap_by_relation=weighted,within_minus_sibling=weighted['same_leaf']-weighted['siblings'],
          information=info.tolist(),rate=rates.tolist(),half_information_times_gamma=clock,
          bootstrap95_half_gamma=ci.tolist(),
          coarse_first_passage=crossings(parenttrue),fine_first_passage=crossings(leaftrue),
          accepted_candidate_tree=bool(aris[0]>=.9 and aris[1]>=.70 and mass_prior.min()>=.03 and weighted['same_leaf']-weighted['siblings']>=.10),
          runtime_s=round(time.monotonic()-tic,2),
          caveat='A data-trained binary PCA tree is a candidate; acceptance of its structural stability does not prove a canonical Parisi tree; the half-time ordering is not a null-calibrated significance test.')
    fname=f'{model}_seed{seed}_p{npaths}'
    (OUT/(fname+'.json')).write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/(fname+'.npz'),gamma=GAMMA,Y=Y,H=H,leaf_mass=pm,per_path_information=per_path_info)
    print('DONE',model,seed, 'ARI',np.round(aris,2),'gap',round(result['within_minus_sibling'],3),'clock',clock,'pass',result['accepted_candidate_tree'],'seconds',result['runtime_s'],flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['bethe','two_state','product'],default='bethe');parser.add_argument('--seed',type=int,default=4);parser.add_argument('--paths',type=int,default=72)
    args=parser.parse_args();analyze(args.model,args.seed,args.paths)
