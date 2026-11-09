"""Stage 5D: locked, nonselective extraction of candidate hierarchy transients.

Reads previous E5B and E5C results; never simulates a new observation or re-fits a tree.
Stage5C flags are REPEATABILITY gates, not equilibrium certificates.
"""
from __future__ import annotations
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'code')]
from e5_largeN_importance import load
from e3b_e1_heldout_hierarchy import train_tree,predict
E5B=ROOT/'results'/'E5B'; E5C=ROOT/'results'/'E5C';E=ROOT/'results'/'E5D'; F=E/'figures'
F.mkdir(exist_ok=True,parents=True)
RNG=np.random.default_rng(20261018)


def entropy(p):
    p=np.maximum(np.asarray(p,dtype=float),1e-300)
    return -(p*np.log(p)).sum(axis=-1)


def first_passage(p,g,up=.9,down=.8):
    idx=np.flatnonzero(p>=up)
    if not len(idx):return None,False,0
    i=int(idx[0]); rec=np.flatnonzero(p[i+1:]<down)
    if not len(rec):return float(g[i]),False,0
    below=i+1+rec[0]
    # Number of *distinct recross episodes* below the reset threshold after commitment.
    active=False; n=0
    for j in range(i+1,len(g)):
        if not active and p[j]<down:n+=1;active=True
        if active and p[j]>=up:active=False
    return float(g[i]),True,int(n)


def reconstruct(n,seed):
    z=np.load(E5B/f'N{n}_bethe_seed{seed}_p24_P768.npz');
    g=z['gamma'];Y=z['Y'];orig=z['smc'].copy();m=orig.mean(axis=1)
    # max-population variation across *the active independent populations* at each time.
    uncert=.5*np.abs(orig[:,0]-orig[:,1]).sum(-1)
    repaired=np.zeros(len(Y),bool); unresolved=np.zeros(len(Y),bool);
    anchor_count=np.zeros(len(Y),int); pt_gates=np.zeros(len(Y),int)
    cohort=json.loads((E5C/'first_breach'/'LOCKED_COHORT.json').read_text())['cases']
    for i,c in enumerate(cohort):
        if (c['N'],c['seed'])!=(n,seed) or c['kind']!='first-breach':continue
        p=int(c['path']);js=json.loads((E5C/'first_breach'/f'case_{i:03d}.json').read_text())
        dat=np.load(E5C/'first_breach'/f'case_{i:03d}.npz')
        assert np.allclose(dat['gamma'],g[np.searchsorted(g,dat['gamma'][0]):])
        assert js['reference']['gate'];pt_gates[p]+=1
        g0=float(c['gamma']);s=int(np.searchsorted(g,g0)); k=np.asarray(dat['anchored_masses'])
        assert k.shape[1]==len(g)-s
        m[p,s:]=k.mean(axis=0)
        uncert[p,s:]=max_pairwise_tv(k)
        anchor_count[p]=1
        pass1=bool(js['anchor']['pass_all_times'])
        if not pass1:
            path2=E5C/'first_breach'/'second_anchor'/f'second_{i:03d}'
            if path2.with_suffix('.npz').exists():
                j2=json.loads(path2.with_suffix('.json').read_text()); zz=np.load(path2.with_suffix('.npz'))
                assert j2['second_PT']['gate'];pt_gates[p]+=1
                s2=int(np.searchsorted(g,float(zz['gamma'][0]))); assert s2>=s
                k2=np.asarray(zz['masses']);m[p,s2:]=k2.mean(axis=0);uncert[p,s2:]=max_pairwise_tv(k2)
                anchor_count[p]=2;pass1=bool(j2['second_anchor']['passes'])
        repaired[p]=pass1; unresolved[p]=not pass1
    # path never flagged in cohort passes original two-pop TV, use original mean.
    return g,Y,m,uncert,repaired,unresolved,anchor_count,pt_gates,orig


def max_pairwise_tv(a):
    return np.max([.5*np.abs(a[i]-a[j]).sum(-1) for i in range(len(a)) for j in range(i+1,len(a))],axis=0)


def make_metrics(g,Y,m,prior,tree,err,usable):
    _,lab=predict(Y,tree); lab=np.asarray(lab,dtype=int)
    co=m.reshape(len(Y),len(g),2,2).sum(-1)
    cp=co[np.arange(len(Y)),:,lab//2]
    fp=m[np.arange(len(Y)),:,lab]
    ev=[]
    for p in range(len(Y)):
        a=[]
        for z in (cp[p],fp[p]):
            t,r,count=first_passage(z,g)
            a.append(dict(first90=t,recross80=r,recross_episodes=count,terminal=float(z[-1]),max=float(z.max())))
        ev.append(dict(path=p,valid=bool(usable[p]),posterior_population_max_tv=float(err[p].max()),coarse=a[0],fine=a[1],order_gap=(a[1]['first90']-a[0]['first90'] if a[0]['first90'] is not None and a[1]['first90'] is not None else None)))
    ec=entropy(co).mean(0);el=entropy(m).mean(0)
    pc=np.asarray(prior).reshape(2,2).sum(-1)
    I0=entropy(pc)-ec
    I1=entropy(prior)-entropy(pc)-el+ec
    # Bias at time zero remains visible: never shift/subtract by fiat.
    curves=np.stack([I0,I1],axis=-1)
    half=[]
    for k in range(2):
        idx=np.flatnonzero(curves[:,k]>=.5*curves[-1,k])
        half.append(float(g[idx[0]]) if len(idx) and curves[-1,k]>0 else None)
    return ev,curves,half


def boot_half_curves(g,m,pi,b=1000):
    n=len(m);r=np.empty((b,2));r[:]=np.nan
    for bi in range(b):
        a=m[RNG.integers(0,n,n)]
        coarse=a.reshape(n,len(g),2,2).sum(-1)
        pc=pi.reshape(2,2).sum(-1)
        c=entropy(pc)-entropy(coarse).mean(0)
        f=entropy(pi)-entropy(pc)-entropy(a).mean(0)+entropy(coarse).mean(0)
        for i,v in enumerate((c,f)):
            idx=np.flatnonzero(v>=.5*v[-1]) if v[-1]>0 else []
            if len(idx):r[bi,i]=g[idx[0]]
    return [[float(v) for v in np.quantile(r[np.isfinite(r[:,k]),k],[.025,.5,.975])] if np.isfinite(r[:,k]).sum()>=b//2 else None for k in (0,1)]


def main():
    groups=[];cohortrows=[];detailed=[]
    for n in (32,64):
        for seed in (4,5,8):
            g,Y,m,err,repaired,unresolved,counts,pt_gates,orig=reconstruct(n,seed)
            assert len(Y)==24
            a,_=load(n,seed,0)
            tree=train_tree(np.r_[a[:,:8].reshape(-1,n),-a[:,:8].reshape(-1,n)])
            pi=np.asarray(json.loads((E5B/f'N{n}_bethe_seed{seed}_p24_P768.json').read_text())['prior_leaf_mass'])
            valid=~unresolved
            ev,cur,half=make_metrics(g,Y,m,pi,tree,err,np.ones(len(Y),bool))
            eva,cura,halfa=make_metrics(g,Y[valid],m[valid],pi,tree,err[valid],np.ones(valid.sum(),bool))
            # Bootstraps condition on this particular disorder, *not* between-disorder uncertainty.
            bo=boot_half_curves(g,m[valid],pi)
            ni=[]
            for level in ('coarse','fine'):
                v=[x[level] for x in ev];w=[x[level] for x in eva]
                ni.append(dict(level=level,commit_all=sum(e['first90'] is not None for e in v),commit_accepted=sum(e['first90'] is not None for e in w),
                    recross_all=sum(e['recross80'] for e in v),recross_accepted=sum(e['recross80'] for e in w),
                    episodes_all=sum(e['recross_episodes'] for e in v),
                    median_first_accepted=float(np.median([e['first90'] for e in w if e['first90'] is not None])) if any(e['first90'] is not None for e in w) else None))
            groups.append(dict(N=n,seed=seed,paths=24,accepted=int(valid.sum()),unresolved_paths=np.flatnonzero(unresolved).tolist(),repaired_paths=np.flatnonzero(repaired).tolist(),
                one_anchor=int((counts==1).sum()),two_anchors=int((counts==2).sum()),
                half_times_all=half,half_times_accepted=halfa,bootstrap_within_disorder_accepted=bo,
                entropy_curve_at_gamma0_all=cur[0].tolist(),entropy_curve_at_gamma0_accepted=cura[0].tolist(),
                terminal_acquired_info_all=cur[-1].tolist(),terminal_acquired_info_accepted=cura[-1].tolist(),
                event_summary=ni,mean_max_interpopulation_TV=float(err.max(axis=1).mean())))
            for e in ev:cohortrows.append(dict(N=n,seed=seed,**e))
            detailed.append(dict(n=n,seed=seed,g=g,curve=cur,curve_acc=cura,mass=m,valid=valid,Y=Y,pi=pi))
    # Existing exact N16 and exact null controls: class hierarchy is trained per model, not guaranteed stable.
    controls=[]
    for model in ('bethe','two_state','product'):
        seeds=(1,3,4,5,8) if model=='bethe' else (4,)
        for seed in seeds:
            ff=E5B/f'N16_{model}_seed{seed}_p{12 if model=="bethe" else 8}_P{1024}.npz'
            if not ff.exists():continue
            z=np.load(ff);g=z['gamma'];Y=z['Y'];m=z['exact'];
            # derive true leaf labels indirectly from original solver/training pipeline if possible
            # Exact entropy clock needs only the class probs and prior masses at gamma=0.
            pc=m[:,0,:].mean(0);co=m.reshape(len(Y),len(g),2,2).sum(-1)
            ec=entropy(pc.reshape(2,2).sum(-1))-entropy(co).mean(0)
            ef=entropy(pc)-entropy(pc.reshape(2,2).sum(-1))-entropy(m).mean(0)+entropy(co).mean(0)
            cs=np.stack([ec,ef],axis=1)
            hc=[]
            for k in range(2):
                ii=np.flatnonzero(cs[:,k]>=.5*cs[-1,k]) if cs[-1,k]>0 else []
                hc.append(float(g[ii[0]]) if len(ii) else None)
            # Peak posterior leaf commitment and recrossings are partition-dependent, but not endpoint-label-dependent.
            # Without recomputing the exact planted hierarchy, use maximum-mass class as a descriptive quantity only.
            controls.append(dict(model=model,seed=seed,paths=len(Y),half_times=hc,terminal_information=cs[-1].tolist(),
                                 note='Exact posterior; partitions algorithmically inferred, null fine partition not reproducible in Stage5.'))
    rows=groups
    doc=dict(stage='5D',source='Stage5B + Stage5C frozen arrays, no new fields or refitted hierarchies',
         acceptance_definition='repeatability <= 0.10 at every saved time after up to two PT anchors, NOT fully independent posterior certification',
         primary_statistical_unit='three independent disorder realizations per size; within-disorder path bootstrap is conditional and not a disorder CI',
         event_definition='first sampled time when TRUE planted class posterior >=0.9; recrossing any later recorded time below 0.8; right-censored at gamma=3.2; no interpolation',
         groups=groups,controls_exact_n16=controls,
         unresolved=[x for x in cohortrows if not x['valid']],
         scientific_warning='First coarse vs fine 90 percent crossing order is tautologically nested; evidence is in nonzero gaps, information curves and validated overlap structure, not raw ordering.',
         all_path_analysis='Includes both unresolved trajectories for sensitivity; accepted-only analysis excludes them but does not establish independent correctness of remaining paths.')
    (E/'STAGE5D_METRICS.json').write_text(json.dumps(doc,indent=2))
    (E/'STAGE5D_PATH_EVENTS.json').write_text(json.dumps(cohortrows,indent=2))
    # Figure 1 mean conditional information with side-by-side sensitivity.
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.4),sharey=True)
    for ax,n in zip(axes,(32,64)):
        sub=[d for d in detailed if d['n']==n];g=sub[0]['g'];
        for k,label in enumerate(('sector','first glassy level')):
            y=np.stack([d['curve_acc'][:,k] for d in sub]); avg=y.mean(0)
            ln,=ax.plot(g,avg,marker='.',lw=1.8,label=f'{label}, repeat-consistent trajectories')
            lo,hi=np.min(y,axis=0),np.max(y,axis=0)
            ax.fill_between(g,lo,hi,color=ln.get_color(),alpha=.12,label=f'{label}, range over the three instances' if n==32 else None)
            allmean=np.mean([d['curve'][:,k] for d in sub],axis=0)
            ax.plot(g,allmean,ls=':',color=ln.get_color(),lw=1.2)
        ax.set(title=f'({"ab"[n==64]}) Bethe glasses, N={n}, three instances',xlabel=r'internal clock $\gamma$',xlim=(0,3.2))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('information about the level (nats)');axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(F/'E5D_information_sensitivity.png',dpi=170);fig.savefig(F/'E5D_information_sensitivity.pdf');plt.close(fig)
    # Figure 2 event prevalence, no selection based on favorable events.
    fig,axs=plt.subplots(1,2,figsize=(10.7,4.5),sharey=True)
    for ax,n in zip(axs,(32,64)):
        xs=[r for r in groups if r['N']==n];pos=np.arange(3);wd=.18
        for j,(level,field) in enumerate((('coarse','commit_accepted'),('fine','commit_accepted'),('coarse','recross_accepted'),('fine','recross_accepted'))):
            values=[r['event_summary'][0 if level=='coarse' else 1][field]/r['accepted'] for r in xs]
            ax.bar(pos+(j-1.5)*wd,values,width=wd,label=f'{level} {"90% hit" if "commit" in field else "recross <80%"}')
        ax.set(xticks=pos,xticklabels=['seed 4','seed 5','seed 8'],ylim=(0,1),title=f'N={n}, accepted-path fraction');ax.grid(axis='y',alpha=.2)
    axs[0].legend(fontsize=7);fig.tight_layout();fig.savefig(F/'E5D_commitments_recrossings.png',dpi=170);fig.savefig(F/'E5D_commitments_recrossings.pdf');plt.close(fig)
    # Figure 3 path-level start-to-end uncertainty at anchor with worst disagreement shading.
    q=[d for d in detailed if d['n']==64 and d['seed']==5][0]
    fig,ax=plt.subplots(figsize=(8,4.7));sub=[r for r in cohortrows if r['N']==64 and r['seed']==5]
    x=np.arange(len(sub)); val=[r['posterior_population_max_tv'] for r in sub]
    ax.bar(x,val);ax.axhline(.10,linestyle='--',color='black',lw=1,label='0.10 repeatability gate');
    for r in sub:
        if not r['valid']:ax.scatter([r['path']],[r['posterior_population_max_tv']],marker='x',s=80,color='red',zorder=3)
    ax.set(xlabel='Frozen path index',ylabel='Worst pairwise posterior-class TV over time',title='N=64 disorder 5: residual failures retained');ax.legend();fig.tight_layout();fig.savefig(F/'E5D_pathwise_uncertainty.png',dpi=170);fig.savefig(F/'E5D_pathwise_uncertainty.pdf');plt.close(fig)
    print('GROUPS:')
    for r in groups: print(r['N'],r['seed'], 'accepted',r['accepted'],'half',r['half_times_accepted'], 'events',r['event_summary'])
    print('Exact control times',controls)
    print('Wrote', E)
    return doc
if __name__=='__main__':main()
