"""Stage 3B: held-out-data-derived candidate nested spin hierarchies (NOT established pure states).

Train deterministic covariance-split trees on two INDEPENDENT equilibrium reference pools.
Evaluate class information and exact Gaussian-channel rate on fresh planted fields,
with exhaustive N=16 posterior sums and true class priors. Check partition stability
with adjusted Rand (invariant under class label permutation) on a third held-out pool.
"""
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import pidlib as P
GAM=np.array([0,.003,.008,.015,.025,.05,.075,.10,.15,.25,.4,.65,1.,1.6,2.5,4.,7.,10.])
M=128

def split_vec(Y):
    mu=np.mean(Y,axis=0);C=(Y-mu).T@(Y-mu)/len(Y)
    _,v=np.linalg.eigh(C);v=v[:,-1]
    # canonical sign eliminates unnecessary sign switching
    if v[np.argmax(abs(v))]<0:v=-v
    s=Y@v
    threshold=float(np.median(s))
    if np.min(s)>=threshold or np.max(s)<=threshold:threshold=float((np.min(s)+np.max(s))/2)
    return v,threshold

def train_tree(train):
    v0,t0=split_vec(train)
    cl=(train@v0>t0).astype(int)
    sub=[]
    for i in range(2):
        dat=train[cl==i]
        if len(dat)<32:
            sub.append((np.zeros(train.shape[1]),1e12))
        else:sub.append(split_vec(dat))
    return v0,t0,sub

def predict(Y,tree):
    v0,t0,sub=tree
    a=(Y@v0>t0).astype(int)
    b=np.zeros(len(a),int)
    for k,(v,t) in enumerate(sub):
        ids=np.where(a==k)[0]
        if len(ids):b[ids]=(Y[ids]@v>t).astype(int)
    return a,2*a+b

def adjusted_rand(c,d):
    _,a=np.unique(c,return_inverse=True);_,b=np.unique(d,return_inverse=True)
    mat=np.zeros((max(a)+1,max(b)+1),np.int64)
    np.add.at(mat,(a,b),1)
    choose2=lambda x: x*(x-1)/2
    total=choose2(len(a));p=choose2(mat).sum();r=choose2(mat.sum(1)).sum();s=choose2(mat.sum(0)).sum()
    expected=r*s/total if total else 0;denom=.5*(r+s)-expected
    return float((p-expected)/denom) if denom>1e-12 else 1.0

def run(kind,seed):
    path=ROOT/'results'/'E3B_instances'/f'{kind}_seed{seed}.json'
    if path.exists():return json.loads(path.read_text())
    rng=np.random.default_rng(21000+10*seed+(0 if kind=='bethe' else 10000))
    inst=P.bethe_spin_glass(16,3,2.5,'gauss',seed) if kind=='bethe' else P.planar_ea(4,2.5,'gauss',seed)
    S=P.ExactSolver(inst);X=S.X.astype(np.float64);mu=np.exp(S.logmu)
    T1=train_tree(S.sample_prior(3000,rng));T2=train_tree(S.sample_prior(3000,rng));Yhold=S.sample_prior(1600,rng)
    c1,c2=predict(X,T1);a1,a2=predict(Yhold,T1);b1,b2=predict(Yhold,T2)
    ar1,ar2=adjusted_rand(a1,b1),adjusted_rand(a2,b2)
    # labels m_1, m_2 as group IDs, each mapping any full configuration to one leaf
    groups=[];prior=[]
    for ids in (c1,c2):
        sub=[np.where(ids==k)[0] for k in range(2 if ids is c1 else 4)]
        groups.append(sub)
        prior.append(np.array([mu[v].sum() for v in sub]))
    xgroups=[[X[ix].copy() for ix in grp] for grp in groups]
    Y=S.sample_prior(M,rng);Z=rng.standard_normal(Y.shape)
    lab_true=predict(Y,T1)
    out={'kind':kind,'seed':seed,'N':16,'n_eval_fields':M,'training_sample_count_each':3000,
         'heldout_stability_count':1600,'ari_coarse':ar1,'ari_fine':ar2,'class_prior_masses':list(map(lambda a:a.tolist(),prior)),
         'gamma':GAM.tolist(),'class_info':[],'class_rate':[],
         'provenance':'held-out PCA recursive bisection; candidate classes, not proven Parisi states'}
    for g in GAM:
        H=g*Y+np.sqrt(g)*Z
        rates=np.empty((M,2));infos=np.empty((M,2))
        for i,h in enumerate(H):
            lp=S.logmu+h@X.T;w=np.exp(lp-logsumexp(lp));m=w@X
            for lev in (0,1):
                cls=groups[lev]
                p=np.array([w[idx].sum() for idx in cls]);v=np.array([w[idx]@xgroups[lev][j] for j,idx in enumerate(cls)]);
                r=.5*(np.sum(np.sum(v*v,axis=1)/np.maximum(p,1e-300))-m@m)
                ix=lab_true[lev][i]
                infos[i,lev]=np.log(max(p[ix],1e-300)/max(prior[lev][ix],1e-300))
                rates[i,lev]=max(0.,r)
        rinc=np.c_[rates[:,0],rates[:,1]-rates[:,0]]
        iinc=np.c_[infos[:,0],infos[:,1]-infos[:,0]]
        out['class_info'].append(iinc.mean(0).tolist())
        out['class_rate'].append(rinc.mean(0).tolist())
    oI=np.array(out['class_info']);oR=np.array(out['class_rate']);out['rate_integral_nats']=np.trapezoid(oR,GAM,axis=0).tolist()
    out['information_at_gamma10']=oI[-1].tolist()
    out['rate_integral_minus_terminal_information']=(np.trapezoid(oR,GAM,axis=0)-oI[-1]).tolist()
    path.write_text(json.dumps(out,indent=2))
    print(f"{kind} {seed} ARI coarse/fine={ar1:.2f}/{ar2:.2f} integral gap {np.round(out['rate_integral_minus_terminal_information'],3)}",flush=True)
    return out

def main():
    tic=time.time();d=ROOT/'results'/'E3B_instances';d.mkdir(exist_ok=True)
    res=[run(kind,seed) for kind in ('bethe','planar') for seed in (1,2,3)]
    out={'gamma':GAM.tolist(),'nfields':M,'ntrain':3000,'by_family':{},'time_sec':time.time()-tic,
         'interpretation':'Held-out tree stability is essential; rate identity valid for any trained partition, not a Parisi proof.'}
    for kind in ('bethe','planar'):
        r=[q for q in res if q['kind']==kind]
        out['by_family'][kind]={k:[x[k] for x in r] for k in ('ari_coarse','ari_fine','rate_integral_minus_terminal_information','class_prior_masses')}
        out['by_family'][kind]['mean_level_rate']=np.mean(np.array([x['class_rate'] for x in r]),axis=0).tolist()
        out['by_family'][kind]['mean_level_info']=np.mean(np.array([x['class_info'] for x in r]),axis=0).tolist()
    (ROOT/'results'/'E3B_e1_heldout_hierarchy_summary.json').write_text(json.dumps(out,indent=2))
    print('DONE',round(time.time()-tic,1),'s',flush=True)
if __name__=='__main__':
    if len(sys.argv)==3: run(sys.argv[1],int(sys.argv[2]))
    else: main()
