"""Stage 5 finite-N PID transients on independent prior Monte Carlo particle libraries.
Posterior information is APPROXIMATE and is masked by nominal particle ESS;
there is no claim that temporal correlation/rare basins are captured by nominal ESS.
"""
from pathlib import Path
import json,sys, argparse
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'experiments')]
from e3b_e1_heldout_hierarchy import train_tree,predict,adjusted_rand
RESULT=ROOT/'results'/'E5F'/'importance';RESULT.mkdir(parents=True,exist_ok=True)
GAM=np.array([0,.002,.004,.007,.010,.016,.025,.04,.06,.09,.13,.2,.3,.5])

def load(N,seed,repeat):
    f=ROOT/'results'/'E5F'/'prior'/f'prior_N{N}_disorder{seed}_repeat{repeat}_sw12000.npz'
    v=np.load(f);return v['chains'].astype(float),v['roundtrips'].astype(int)

def entropy(p):return -np.sum(p*np.log(np.maximum(p,1e-300)),axis=-1)

def run(N,seed,paths=96):
    a,ta=load(N,seed,0);b,tb=load(N,seed,1)
    # Both train datasets each come from a different independent MC run.
    xa=a[:,:8].reshape(-1,N);xb=b[:,:8].reshape(-1,N)
    # Held-out chain pool does not enter either tree fit.
    hold=a[:,8:].reshape(-1,N)
    # The posterior particle pool is drawn from the OTHER independent MC run.
    particles=b[:,8:].reshape(-1,N)
    trA=train_tree(np.r_[xa,-xa]);trB=train_tree(np.r_[xb,-xb]);
    aa,af=predict(hold,trA);ba,bf=predict(hold,trB)
    ari=[adjusted_rand(aa,ba),adjusted_rand(af,bf)]
    cm,fm=predict(particles,trA)
    member=(fm[:,None]==np.arange(4)).astype(float)
    pi=member.mean(axis=0);coarse_pi=pi.reshape(2,2).sum(axis=1)
    priors_train=predict(xa,trA)[1]
    pi_train=np.bincount(priors_train,minlength=4)/len(priors_train)
    prdiscrep=np.max(np.abs(pi-pi_train))
    # Sample new endpoints from holdout (different chains from particle pool).
    rng=np.random.default_rng(882417+N*100+seed)
    Y=hold[rng.choice(len(hold),size=paths,replace=True)]
    yc,yf=predict(Y,trA)
    h=np.zeros((paths,N));last=0
    masses=[]; info=[];rates=[];essarr=[];per_path=[]
    for g in GAM:
        d=g-last
        if d>0:h+=d*Y+np.sqrt(d)*rng.standard_normal((paths,N))
        last=g
        lw=h@particles.T
        w=np.exp(lw-logsumexp(lw,axis=1,keepdims=True))
        e=1./np.sum(w*w,axis=1)
        leaf=w@member;par=leaf.reshape(paths,2,2).sum(axis=-1)
        m=w@particles
        leafmom=np.stack([(w*member[:,j])@particles for j in range(4)],axis=1)
        pmom=leafmom.reshape(paths,2,2,N).sum(axis=2)
        E1=(np.sum(pmom*pmom,axis=-1)/np.maximum(par,1e-200)).sum(-1)
        E2=(np.sum(leafmom*leafmom,axis=-1)/np.maximum(leaf,1e-200)).sum(-1)
        rat=np.stack([.5*(E1-(m*m).sum(-1)),.5*(E2-E1)],axis=1)
        mco=entropy(coarse_pi)-entropy(par)
        mfi=entropy(pi)-entropy(coarse_pi)-entropy(leaf)+entropy(par)
        per_path.append(np.stack([mco,mfi],axis=1));info.append([mco.mean(),mfi.mean()])
        rates.append(rat.mean(0));essarr.append(e);masses.append(leaf)
    essarr=np.array(essarr);masses=np.array(masses);info=np.array(info);rates=np.array(rates)
    gate=np.maximum(100,.05*len(particles))
    flags=(essarr>=gate).mean(axis=1)
    # Require *all* evaluated observations above ESS threshold for group-average inference.
    good=np.flatnonzero(flags==1.)
    last_good=good[-1] if len(good) and np.array_equal(good,np.arange(good[-1]+1)) else -1
    validated=[float(GAM[i]) for i in good]
    labeltrue=masses[np.arange(len(GAM))[:,None],np.arange(paths)[None,:],yf[None,:]]
    parmass=masses.reshape(len(GAM),paths,2,2).sum(-1)
    partrue=parmass[np.arange(len(GAM))[:,None],np.arange(paths)[None,:],yc[None,:]]
    def first_pass(p):
        valid=p[:last_good+1] if last_good>=0 else np.empty((0,paths))
        if not len(valid):return dict(validated=False)
        hits=valid>=.90;ever=hits.any(0);idx=hits.argmax(0)
        rec=np.array([np.any(valid[idx[i]+1:,i]<.80) if ever[i] else False for i in range(paths)])
        return dict(validated=True,hit_fraction=float(ever.mean()),recross_fraction_all=float(rec.mean()),
           median_hit_gamma=float(np.median(GAM[idx[ever]])) if ever.any() else None)
    # A prior overlap-gap estimate: Monte Carlo sample mean spins in each class.
    mm=np.stack([particles[fm==j].mean(0) if np.any(fm==j) else np.zeros(N) for j in range(4)])
    Q=mm@mm.T/N
    def qavg(typ):
        v=[];w=[]
        for i in range(4):
            for j in range(4):
                k='same_leaf' if i==j else ('siblings' if i//2==j//2 else 'different_parents')
                if k==typ:v.append(Q[i,j]);w.append(pi[i]*pi[j])
        return float(np.average(v,weights=w)) if sum(w)>0 else None
    gap=qavg('same_leaf')-qavg('siblings')
    treepass=bool(ari[0]>=.90 and ari[1]>=.70 and pi.min()>=.03 and gap>=.10 and prdiscrep<=.15)
    res=dict(N=N,disorder=seed,paths=paths,prior_roundtrips_min=[int(ta.min()),int(tb.min())],
        independent_prior_runs=2,per_run_sample_shape=list(a.shape),training_samples_per_independent_run=len(xa),
        heldout_sample_count=len(hold),posterior_reference_particles=len(particles),
        nominal_posterior_ESS_gate=float(gate),
        heldout_ARI=ari,prior_leaf_mass_reference=pi.tolist(),prior_leaf_mass_training=pi_train.tolist(),max_prior_mass_disagreement=float(prdiscrep),
        prior_overlap_by_relation={'same_leaf':qavg('same_leaf'),'siblings':qavg('siblings'),'different_parents':qavg('different_parents')},
        overlap_gap=float(gap),structural_candidate_pass=treepass,
        gamma=GAM.tolist(),information=info.tolist(),information_rate=rates.tolist(),
        ESS_min_over_fields=essarr.min(1).tolist(),ESS_median_over_fields=np.median(essarr,axis=1).tolist(),
        ESS_fraction_fields_accepted=flags.tolist(),last_all_fields_ESS_qualified_gamma=float(GAM[last_good]) if last_good>=0 else None,
        posterior_true_coarse_first_passage=first_pass(partrue),posterior_true_leaf_first_passage=first_pass(labeltrue),
        inference_type='prior-library importance posterior, NOT exact posterior or fully validated MCMC at all observation fields',
        limitation='ESS is nominal for temporally correlated tempering draws and cannot diagnose absent metastable valleys. No half-information clocks from incomplete gamma windows.')
    path=RESULT/f'importance_N{N}_disorder{seed}_p{paths}'
    (path.with_suffix('.json')).write_text(json.dumps(res,indent=2))
    np.savez_compressed(path.with_suffix('.npz'),gamma=GAM,Y=Y,masses=masses,ESS=essarr,information=info)
    print('IMPORTANCE',N,seed,'ARI',np.round(ari,2),'gap',round(gap,3),'prior diff',round(prdiscrep,3),'tree',treepass,
          'last ESS gamma',res['last_all_fields_ESS_qualified_gamma'],'min round trips',res['prior_roundtrips_min'],flush=True)
    return res
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('N',type=int);p.add_argument('seed',type=int);p.add_argument('--paths',type=int,default=80);a=p.parse_args();run(a.N,a.seed,a.paths)
