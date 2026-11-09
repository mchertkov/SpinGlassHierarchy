"""Stage 5C: frozen-field two-start PT cross-check and PT-anchored path extension.

Use frozen Stage5B fields only. Does not claim PT supplies an exact posterior oracle.
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'experiments')]
import pidlib as P
from e5_largeN_importance import load
from e3b_e1_heldout_hierarchy import train_tree,predict
from e4de_independent_heatbath import initial
from e4dg_numba_kernel import simulate
from e5b_sequential_smc import SequentialSMC,GRID

CASES=[(32,4,17,.46,'difficult'),(64,5,18,.36,'difficult'),
       (32,5,1,.36,'control'),(32,8,6,.58,'control'),
       (64,4,1,.36,'control'),(64,8,6,.58,'control')]
DIR=ROOT/'results'/'E5C';DIR.mkdir(parents=True,exist_ok=True)

def fixed_field(N,disorder,path,gamma):
    zz=np.load(ROOT/'results'/'E5B'/f'N{N}_bethe_seed{disorder}_p24_P768.npz')
    gi=int(np.argmin(abs(zz['gamma']-gamma)))
    assert abs(zz['gamma'][gi]-gamma)<1e-10
    a,_=load(N,disorder,0)
    tree=train_tree(np.r_[a[:,:8].reshape(-1,N),-a[:,:8].reshape(-1,N)])
    return tree,zz['H'][path,gi].copy(),zz['H'][path].copy(),zz['smc'][path,:,gi].copy(),gi

def one_pt(N,disorder,h,tree,seed,sweeps=24000,stride=20):
    inst=P.bethe_spin_glass(N,3,2.5,'gauss',disorder)
    C=6;R=28
    betas=2.5*np.linspace(.000001,1,R)**1.15
    fields=np.repeat(h[None],C,axis=0)
    rng=np.random.default_rng(seed)
    x,E=initial(inst,betas,fields,rng,half_ordered=False)
    x[:,3:,:]=1;E=inst.energy(x).astype(float)
    tags=np.broadcast_to(np.arange(R,dtype=np.int32)[:,None],(R,C)).copy()
    seen=np.zeros(C,dtype=np.bool_);trips=np.zeros(C,np.int64)
    acc=np.zeros(R-1,np.int64);att=np.zeros(R-1,np.int64)
    fa=np.zeros(R,np.int64);fo=np.zeros(R,np.int64)
    colors=np.full((len(inst.colors),max(len(v) for v in inst.colors)),-1,np.int32)
    for ci,cl in enumerate(inst.colors): colors[ci,:len(cl)]=cl
    cold,energies=simulate(x,E,fields,tags,seen,trips,acc,att,fa,fo,
                            betas,colors,inst.nbr,inst.Jn,0,sweeps,sweeps//2,stride,0,False,
                            np.random.default_rng(seed+93719),np.random.default_rng(seed+121))
    assert np.max(np.abs(inst.energy(x)-E))<1e-6
    L=predict(cold.reshape(-1,N),tree)[1].reshape(len(cold),C)
    def mass(arr):return np.bincount(arr.reshape(-1),minlength=4).astype(float)/arr.size
    p=mass(L);start_tv=.5*np.abs(mass(L[:,:3])-mass(L[:,3:])).sum()
    split_tv=.5*np.abs(mass(L[:len(L)//2])-mass(L[len(L)//2:])).sum()
    energy_split=abs(energies[:len(L)//2].mean()-energies[len(L)//2:].mean())/N
    gate=bool(trips.min()>=6 and start_tv<=.1 and split_tv<=.1 and energy_split<=.05)
    return dict(mass=p,roundtrips=trips,start_tv=float(start_tv),split_tv=float(split_tv),
        energy_split_per_spin=float(energy_split),gate=gate,leaf=L,cold=cold,energies=energies)

def report_pt(d):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items() if k not in ('leaf','cold','energies')}

def anchor_extend(N,disorder,tree,H,gi,cold,base_seed,npops=3,particle_count=768):
    inst=P.bethe_spin_glass(N,3,2.5,'gauss',disorder)
    h=H[gi]
    out=[]
    for pop_id in range(npops):
        rng=np.random.default_rng(base_seed+pop_id*1151)
        pool=cold.reshape(-1,N)
        # Distinct draws from retained PT samples; independence is limited by PT autocorrelation.
        select=rng.choice(len(pool),size=particle_count,replace=len(pool)<particle_count)
        obj=SequentialSMC(pool[select],inst,tree,rng,rejuvenation=24)
        obj.h=h.copy();obj.gamma=float(GRID[gi]);obj.weights[:]=1./particle_count
        masses=[obj.mass()]
        for g,hh in zip(GRID[gi+1:],H[gi+1:]):
            obj.assimilate(float(g),hh)
            masses.append(obj.mass())
        out.append(masses)
    M=np.array(out)
    tv=np.array([.5*np.abs(M[i]-M[j]).sum(-1) for i in range(npops) for j in range(i+1,npops)])
    return dict(masses=M,max_pair_tv_by_time=tv.max(axis=0),mean_pair_tv_by_time=tv.mean(axis=0),
                pass_all_times=bool((tv<=.1).all()), worst_pair_tv=float(tv.max()))

def main(case,anchor=True,extra_repeat=False):
    start=time.monotonic();N,dis,path,g,typ=CASES[case]
    tree,h,H,smc,gi=fixed_field(N,dis,path,g)
    seed=523177+case*95111
    d=one_pt(N,dis,h,tree,seed)
    rpt=dict(case=case,N=N,disorder=dis,path=path,gamma=g,kind=typ,sweeps=24000,
            original_smc_masses=smc.tolist(),pt=report_pt(d),
            pt_to_original_smc_TV=[float(.5*abs(p-d['mass']).sum()) for p in smc],
            observation_hash=__import__('hashlib').sha256(h.tobytes()).hexdigest(),
            note='Operational PT gate is necessary but not sufficient for equilibrium')
    out=DIR/('N%d_seed%d_path%d_g%s' % (N,dis,path,(f'{g:.2f}').replace('.','p')))
    np.savez_compressed(out.with_suffix('.npz'),H=H,h=h,cold=d['cold'],energies=d['energies'],leaf=d['leaf'])
    if typ=='difficult' or extra_repeat:
        d2=one_pt(N,dis,h,tree,seed+240889)
        rpt['independent_pt']=report_pt(d2)
        rpt['pt_repeat_TV']=float(.5*abs(d2['mass']-d['mass']).sum())
        np.savez_compressed(DIR/(out.name+'_independent.npz'),cold=d2['cold'],leaf=d2['leaf'])
    if anchor and d['gate']:
        a=anchor_extend(N,dis,tree,H,gi,d['cold'],seed+4391)
        rpt['anchor']=dict(max_pair_tv_by_time=a['max_pair_tv_by_time'].tolist(),
                           mean_pair_tv_by_time=a['mean_pair_tv_by_time'].tolist(),
                           worst_pair_tv=a['worst_pair_tv'],pass_all_times=a['pass_all_times'])
        np.savez_compressed(DIR/(out.name+'_anchor.npz'),gamma=GRID[gi:],masses=a['masses'])
    rpt['elapsed_s']=float(time.monotonic()-start)
    out.with_suffix('.json').write_text(json.dumps(rpt,indent=2))
    print('STAGE5C',case,typ,'N',N,'seed',dis,'g',g,'PT gate',d['gate'],'trips',d['roundtrips'].min(),
          'old TV',np.round(rpt['pt_to_original_smc_TV'],3),
          'new anchored max TV',round(rpt.get('anchor',{}).get('worst_pair_tv',-1),3),
          'runtime',round(rpt['elapsed_s'],1),flush=True)
    return rpt
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int,required=True,choices=range(6));ap.add_argument('--no-anchor',action='store_true');args=ap.parse_args()
    main(args.case,anchor=not args.no_anchor)
