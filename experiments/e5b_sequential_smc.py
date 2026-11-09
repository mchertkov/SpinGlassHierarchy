"""Stage 5B: path-consistent sequential Gaussian-observation posterior SMC.

Dynamic finite-size class observations; exact N16 posterior controls. All Ising
configurations ±1; Gibbs kernel targets exp(beta_J H0(y) + h.y).
Output never conflates posterior-weight ESS with effective independent posterior draws.
"""
from __future__ import annotations
from pathlib import Path
import sys, json, argparse, time
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'experiments')]
import pidlib as P
from e3b_e1_heldout_hierarchy import train_tree,predict,adjusted_rand
from e5_exact_transients import setup
OUT=ROOT/'results'/'E5B'; OUT.mkdir(parents=True,exist_ok=True)
# Finely resolved early class information, moderately spaced terminal observations.
GRID=np.array([0,.008,.016,.025,.04,.06,.085,.12,.16,.21,.28,.36,.46,.58,.72,.9,1.15,1.45,1.85,2.4,3.2])

def entropy(p):
    p=np.asarray(p,float)
    return -np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0),axis=-1)

def systematic_resample(weights,rng):
    """Unbiased marginal categorical offspring counts conditional on weights."""
    n=len(weights)
    pts=(rng.random()+np.arange(n))/n
    return np.minimum(np.searchsorted(np.cumsum(weights),pts),n-1)

def heatbath_rejuvenate(X,inst,h,rng,sweeps=12,global_flips=True):
    """A composition of exact Gibbs color-block kernels for one fixed observation h."""
    X=X.copy().astype(np.int8)
    for _ in range(sweeps):
        pad=np.concatenate((X,np.zeros((len(X),1),dtype=np.int8)),axis=1)
        for cls in inst.colors:
            local=np.sum(pad[:,inst.nbr[cls]]*inst.Jn[cls][None,:,:],axis=-1)
            field=inst.beta_J*local+h[cls][None,:]
            pr=.5*(1+np.tanh(field))
            X[:,cls]=np.where(rng.random(pr.shape)<pr,1,-1).astype(np.int8)
            pad[:,cls]=X[:,cls]
        if global_flips:
            loga=-2*(X.astype(float)@h)
            take=np.log(np.maximum(rng.random(len(X)),1e-300))<loga
            X[take]*=-1
    return X

class SequentialSMC:
    def __init__(self,particles,inst,tree,rng,threshold=.70,rejuvenation=12,min_step=.002,oracle_sampler=None):
        self.X=np.asarray(particles,np.int8).copy();self.inst=inst;self.tree=tree
        self.P=len(self.X);self.weights=np.full(self.P,1/self.P)
        self.rng=rng;self.oracle_sampler=oracle_sampler;self.threshold=threshold;self.rejuvenation=rejuvenation;self.min_step=min_step
        self.anc=np.arange(self.P);self.h=np.zeros(inst.N);self.gamma=0.
        self.resamples=0;self.splits=0;self.min_increment_ratio=1.;self.lastess=self.P
    def _forecast(self,dh):
        lw=np.log(np.maximum(self.weights,1e-300))+self.X.astype(float)@dh
        w=np.exp(lw-logsumexp(lw))
        ess=1/np.sum(w*w)
        return w,float(ess)
    def _accept(self,g,h):
        w,ess=self._forecast(h-self.h)
        self.weights=w;self.lastess=ess;self.h=h.copy();self.gamma=g
        self.min_increment_ratio=min(self.min_increment_ratio,ess/self.P)
        if ess < self.threshold*self.P:
            idx=systematic_resample(self.weights,self.rng)
            self.X=self.X[idx].copy();self.anc=self.anc[idx].copy()
            self.weights[:]=1/self.P;self.resamples+=1
            self.X=(self.oracle_sampler.sample(self.h[None,:],self.P,self.rng)[0].astype(np.int8)
                    if self.oracle_sampler is not None else
                    heatbath_rejuvenate(self.X,self.inst,self.h,self.rng,self.rejuvenation))
    def assimilate(self,g1,h1,depth=0):
        if g1<=self.gamma:raise ValueError('non-increasing observation time')
        _,ess=self._forecast(h1-self.h)
        if ess < .58*self.P and (g1-self.gamma)>self.min_step and depth<12:
            start_g=self.gamma;start_h=self.h.copy();dt=g1-start_g
            mid=(start_g+g1)/2
            # Exact Brownian bridge conditioned on the entire coarse observed increment.
            hm=.5*(start_h+h1)+np.sqrt(dt/4)*self.rng.standard_normal(len(self.h))
            self.splits+=1
            self.assimilate(mid,hm,depth+1)
            self.assimilate(g1,h1,depth+1)
        else:self._accept(g1,h1)
    def mass(self):
        _,lab=predict(self.X,self.tree)
        return np.bincount(lab,weights=self.weights,minlength=4)
    def record(self):
        return self.mass(),self.lastess,self.P and len(np.unique(self.anc)),len(np.unique(self.X,axis=0))

def latent_paths(Y,grid,rng):
    # Generate a *single* Gaussian observation process per endpoint. Grid values are kept
    # exactly fixed across independent SMC populations.
    H=np.zeros((len(grid),len(Y)));prev=0.
    for k,g in enumerate(grid[1:],1):
        d=g-prev;H[k]=H[k-1]+d*Y+np.sqrt(d)*rng.standard_normal(len(Y));prev=g
    return H

def smc_path(inst,tree,Y,H,Xinitial,runseed,kwargs=None):
    kwargs=kwargs or {};r=np.random.default_rng(runseed)
    pop=SequentialSMC(Xinitial,inst,tree,r,**kwargs)
    masses=[];ess=[];anc=[];uni=[];rs=[];spl=[];magnets=[];draws=[]
    for g,h in zip(GRID,H):
        if g>0:pop.assimilate(float(g),h)
        m,e,a,u=pop.record()
        masses.append(m);ess.append(e);anc.append(a);uni.append(u);rs.append(pop.resamples);spl.append(pop.splits)
        magnets.append(pop.weights@pop.X.astype(float))
        draws.append(pop.X[pop.rng.choice(pop.P,p=pop.weights)].copy())
    return dict(mass=np.array(masses),ESS=np.array(ess),ancestors=np.array(anc),distinct=np.array(uni),resamples=np.array(rs),splits=np.array(spl),magnet=np.array(magnets),draw=np.array(draws))

def exact_path(solver,tree,H):
    X=solver.X.astype(np.float64);mu=solver.logmu
    _,leaf=predict(X,tree)
    member=(leaf[:,None]==np.arange(4)).astype(float)
    out=[]
    for h in H:
        lw=mu+X@h;w=np.exp(lw-logsumexp(lw))
        out.append(w@member)
    return np.asarray(out)

def exact_magnet_path(solver,H):
    X=solver.X.astype(np.float64);mu=solver.logmu;out=[]
    for h in H:
        w=np.exp(mu+X@h-logsumexp(mu+X@h));out.append(w@X)
    return np.array(out)

def splitstats(m,Y,tree,pi):
    cy,fy=predict(Y[None],tree)
    leaf=m[:,fy[0]]
    coarse=m.reshape(len(m),2,2).sum(-1)[:,cy[0]]
    def first(p):
        ii=np.flatnonzero(p>=.90)
        if not len(ii):return dict(first=None,recross=False)
        j=ii[0];return dict(first=float(GRID[j]),recross=bool(np.any(p[j+1:]<.80)))
    info=np.stack([np.log(np.maximum(coarse,1e-300)/max(pi.reshape(2,2).sum(1)[cy[0]],1e-300)),
       np.log(np.maximum(leaf,1e-300)/max(pi[fy[0]],1e-300))-np.log(np.maximum(coarse,1e-300)/max(pi.reshape(2,2).sum(1)[cy[0]],1e-300))],axis=1)
    return dict(first_coarse=first(coarse),first_fine=first(leaf),information=info)

def run_n16(model='bethe',disorder=4,paths=6,particles=1024,sweeps=16,seed=211):
    t=time.monotonic();solver=setup(model,disorder);inst=solver.inst
    rng=np.random.default_rng(seed+disorder*113+{'bethe':0,'two_state':201,'product':401}[model])
    train=solver.sample_prior(10000,rng);tree=train_tree(np.r_[train,-train]);
    # For nulls the initialization/proposal kernel must target the *null* prior:
    # a Bethe interaction heat-bath kernel is incorrect for null models.
    if model!='bethe':sweeps=0
    # no post-resampling Gibbs for nulls -> SMC may fail; purpose is to expose invalid high gamma.
    _,labs=predict(solver.X,tree);pi=np.bincount(labs,weights=np.exp(solver.logmu),minlength=4)
    Y=solver.sample_prior(paths,rng)
    records=[];mass=[];exact=[];posterior_tv=[];cross=[];magnet_err=[]
    for p,(y) in enumerate(Y):
        H=latent_paths(y,GRID,rng)
        target=exact_path(solver,tree,H);mtarget=exact_magnet_path(solver,H);z=[];run=[]
        for k in range(2):
            X=solver.sample_prior(particles,rng)
            res=smc_path(inst,tree,y,H,X,int(seed+100000*disorder+1000*p+k*43),dict(rejuvenation=sweeps,oracle_sampler=solver if model!='bethe' else None))
            z.append(res['mass']);run.append(res)
        means=np.mean(z,axis=0)
        tv=.5*np.abs(means-target).sum(-1)
        tvmax=float(tv.max());
        magnet_rmse=np.sqrt(np.mean((.5*(run[0]['magnet']+run[1]['magnet'])-mtarget)**2))
        magnet_err.append(float(magnet_rmse))
        records.append(dict(path=p,tv_mean=float(tv.mean()),tv_max=tvmax,magnet_rmse=float(magnet_rmse),
          interpopulation_tv_mean=float((.5*np.abs(z[0]-z[1]).sum(-1)).mean()),
          resamples=[int(q['resamples'][-1]) for q in run],
          splits=[int(q['splits'][-1]) for q in run],
          min_ancestors=[int(q['ancestors'].min()) for q in run],
          min_weight_ESS=[float(q['ESS'].min()) for q in run],
          min_distinct=[int(q['distinct'].min()) for q in run],
          true_cross=splitstats(target,y,tree,pi)['first_fine'],
          smc_cross=splitstats(means,y,tree,pi)['first_fine']))
        exact.append(target);mass.append(np.array(z));posterior_tv.append(tv);cross.append(H)
    exact=np.array(exact);mass=np.array(mass)
    all_tv=np.array(posterior_tv);inter=.5*np.abs(mass[:,0]-mass[:,1]).sum(-1)
    total=dict(stage='5B1',model=model,N=16,disorder=disorder,paths=paths,particles=particles,
       sweeps=sweeps,mean_tv=float(all_tv.mean()),worst_tv=float(all_tv.max()),
       mean_interpopulation_tv=float(inter.mean()),mean_magnet_rmse=float(np.mean(magnet_err)),
       pass_n16=bool(all_tv.mean()<.10 and all_tv.max()<.25 and inter.mean()<.10),
       gamma=GRID.tolist(),prior_leaf_masses=pi.tolist(),path_records=records,
       endpoint_source='iid exhaustive prior',field_type='correlated Brownian planted',
       notes=('Null controls use exact-posterior iid rejuvenation oracle (available at N16); no claim of a generic null MCMC algorithm' if model!='bethe' else 'heat-bath rejuvenation at each resampling'),
       wall_seconds=float(time.monotonic()-t))
    out=OUT/f'N16_{model}_seed{disorder}_p{paths}_P{particles}'
    out.with_suffix('.json').write_text(json.dumps(total,indent=2))
    np.savez_compressed(out.with_suffix('.npz'),gamma=GRID,Y=Y,H=np.array(cross),exact=exact,smc=mass,tv=all_tv)
    print('N16',model,disorder,'mean tv',round(total['mean_tv'],4),'worst',round(total['worst_tv'],4),
       'two-run',round(total['mean_interpopulation_tv'],4),'pass',total['pass_n16'],'elapsed',round(total['wall_seconds'],1),flush=True)
    return total

def run_large(N=32,disorder=4,paths=4,particles=640,sweeps=18,seed=521):
    from e5_largeN_importance import load
    tic=time.monotonic();inst=P.bethe_spin_glass(N,3,2.5,'gauss',disorder)
    r=np.random.default_rng(seed+disorder*907)
    a,ta=load(N,disorder,0);b,tb=load(N,disorder,1)
    trA=train_tree(np.r_[a[:,:8].reshape(-1,N),-a[:,:8].reshape(-1,N)])
    trB=train_tree(np.r_[b[:,:8].reshape(-1,N),-b[:,:8].reshape(-1,N)])
    hold=b[:,:8].reshape(-1,N)
    c0,f0=predict(hold,trA);c1,f1=predict(hold,trB)
    aris=[adjusted_rand(c0,c1),adjusted_rand(f0,f1)]
    # The source pools below come from two independent prior PT histories;
    # individual entries within each pool still have Monte Carlo dependence.
    libs=[a[:,8:].reshape(-1,N),b[:,8:].reshape(-1,N)]
    Y=hold[r.choice(len(hold),size=paths,replace=False)]
    pi=np.bincount(predict(libs[1],trA)[1],minlength=4)/len(libs[1])
    meta=[];mass=[];obs=[];replicas=[];magnet_differences=[]
    out=OUT/f'N{N}_bethe_seed{disorder}_p{paths}_P{particles}'
    partial=out.with_name(out.name+'_PARTIAL.npz')
    start_i=0
    if partial.exists():
        previous=np.load(partial,allow_pickle=False)
        meta=json.loads(str(previous['meta_json']))
        mass=list(previous['mass']);obs=list(previous['obs']);replicas=list(previous['replicas'])
        magnet_differences=list(previous['magnet_differences'])
        start_i=len(meta)
        r.bit_generator.state=json.loads(str(previous['rng_state']))
        assert start_i<=paths
        print('RESUMING N',N,'disorder',disorder,'from path',start_i,'/',paths,flush=True)
    for i in range(start_i,paths):
        y=Y[i]
        H=latent_paths(y,GRID,r)
        pair=[];run=[]
        for k,lib in enumerate(libs):
            xs=lib[r.choice(len(lib),size=particles,replace=False)]
            res=smc_path(inst,trA,y,H,xs,int(seed+100001*disorder+i*101+k*7),dict(rejuvenation=sweeps))
            pair.append(res['mass']);run.append(res)
        inter=.5*np.abs(pair[0]-pair[1]).sum(-1)
        mag_rmse=float(np.sqrt(np.mean((run[0]['magnet']-run[1]['magnet'])**2)))
        magnet_differences.append(mag_rmse)
        replicas.append(np.array([run[0]['draw'],run[1]['draw']]))
        meta.append(dict(path=i,repeat_TV_mean=float(inter.mean()),repeat_TV_max=float(inter.max()),repeat_magnet_rmse=mag_rmse,
            by_time_TV=inter.tolist(),resamples=[int(q['resamples'][-1]) for q in run],
            min_ancestors=[int(q['ancestors'].min()) for q in run],
            min_distinct=[int(q['distinct'].min()) for q in run],
            true_leaf_cross=[splitstats(z,y,trA,pi)['first_fine'] for z in pair]))
        mass.append(pair);obs.append(H)
        np.savez_compressed(partial,meta_json=np.array(json.dumps(meta)),
            mass=np.array(mass),obs=np.array(obs),replicas=np.array(replicas),
            magnet_differences=np.array(magnet_differences),rng_state=np.array(json.dumps(r.bit_generator.state)))
        print('N',N,'seed',disorder,'field',i+1,'/',paths,'mean inter TV',round(inter.mean(),3),flush=True)
    mass=np.array(mass);replicas=np.array(replicas);tv=.5*np.abs(mass[:,0]-mass[:,1]).sum(-1)
    counts=[float((tv[:,j]<=.10).mean()) for j in range(len(GRID))]
    last_idx=np.where(np.cumprod(np.array(counts)>=.75)==1)[0]
    last_g=float(GRID[last_idx[-1]]) if len(last_idx) else 0.
    # The planted spin and each replica are exchangeable at fixed gamma in the Bayesian
    # model; report only empirical signed overlap differences, not an equilibrium proof.
    plant_rep0=np.mean(Y[:,None,:]*replicas[:,0],axis=-1)
    plant_rep1=np.mean(Y[:,None,:]*replicas[:,1],axis=-1)
    rep_rep=np.mean(replicas[:,0]*replicas[:,1],axis=-1)
    paired_nishimori_diff=(.5*(plant_rep0+plant_rep1)-rep_rep).mean(axis=0)
    total=dict(stage='5B2' if N==32 else '5B3_pilot',N=N,disorder=disorder,paths=paths,particles=particles,
        sweeps=sweeps,prior_independent_histories=2,prior_roundtrip_min=[int(ta.min()),int(tb.min())],
        heldout_ari=aris,prior_leaf_mass=pi.tolist(),gamma=GRID.tolist(),
        mean_repeat_tv=float(tv.mean()),worst_repeat_tv=float(tv.max()),
        mean_repeat_magnet_rmse=float(np.mean(magnet_differences)),
        nishimori_signed_overlap_difference_by_gamma=paired_nishimori_diff.tolist(),
        repeat_TV_by_gamma=tv.mean(axis=0).tolist(),fraction_fields_TV_le_010=counts,
        last_gamma_with_75pct_fields_TV_le_010=float(last_g),
        full_window_repeat_pass=bool(np.all(np.mean(tv,axis=0)<=.10) and np.mean(tv)<=.10 and len(last_idx)==len(GRID)),
        raw_field_records=meta,
        limitations=['Both SMC runs start from Monte Carlo-correlated prior libraries; independent histories but not IID draws',
          'Independent-run agreement is necessary, not sufficient, for posterior equilibrium',
          'Initial planted endpoint comes from a chain subset of one history (not an independent third history)',
          'Global Gibbs mixing between glass basins is not certified'],wall_seconds=float(time.monotonic()-tic))
    out.with_suffix('.json').write_text(json.dumps(total,indent=2))
    np.savez_compressed(out.with_suffix('.npz'),gamma=GRID,Y=Y,H=np.array(obs),smc=mass,replica=replicas)
    if partial.exists():partial.unlink()
    print('N'+str(N)+' summary',disorder,'mean',round(total['mean_repeat_tv'],4),'last good',last_g,'pass',total['full_window_repeat_pass'],'secs',round(total['wall_seconds'],1),flush=True)
    return total

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--N',type=int,default=16);pa.add_argument('--model',default='bethe');pa.add_argument('--disorder',type=int,default=4);pa.add_argument('--paths',type=int,default=4);pa.add_argument('--particles',type=int,default=800);pa.add_argument('--sweeps',type=int,default=16)
    a=pa.parse_args()
    if a.N==16:run_n16(a.model,a.disorder,a.paths,a.particles,a.sweeps)
    elif a.N in (32,64):run_large(a.N,a.disorder,a.paths,a.particles,a.sweeps)
    else:raise ValueError('N must be 16, 32, or 64')
