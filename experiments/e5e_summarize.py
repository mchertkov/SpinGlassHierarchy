"""Frozen Stage 5E analysis: structural+dynamical comparisons without p-value fishing."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'/'E5E';F=R/'figures';F.mkdir(exist_ok=True,parents=True)
rows=[]
for model in ('product','two_state'):
    for N in (32,64):
        for seed in (4,5,8):
            z=json.loads((R/f'{model}_N{N}_rep{seed}.json').read_text())
            rows.append(dict(model=model,N=N,seed=seed,
                coarse_ARI=z['ARI_coarse'],fine_ARI=z['ARI_fine'],
                min_leaf_mass=min(z['prior_leaf_masses']),
                overlap_gap=z['within_minus_sibling_overlap'],
                structural_pass=z['accepted_structural'],
                half_coarse=z['information_half_times'][0],half_fine=z['information_half_times'][1],
                coarse_commit=z['coarse_commit_count'],fine_commit=z['fine_commit_count'],
                fine_recross=z['fine_recross_count'],mean_mc_tv=z['mean_mc_population_tv']))
for N in (32,64):
    for seed in (4,5,8):
        st=json.loads((ROOT/'results'/'E5F'/'importance'/f'importance_N{N}_disorder{seed}_p96.json').read_text())
        d=json.loads((ROOT/'results'/'E5D'/'STAGE5D_METRICS.json').read_text())
        grp=next(x for x in d['groups'] if x['N']==N and x['seed']==seed)
        coarse,fine=grp['half_times_accepted']
        rows.append(dict(model='bethe',N=N,seed=seed,coarse_ARI=st['heldout_ARI'][0],fine_ARI=st['heldout_ARI'][1],
            min_leaf_mass=min(st['prior_leaf_mass_reference']),overlap_gap=st['overlap_gap'],
            structural_pass=st['structural_candidate_pass'],half_coarse=coarse,half_fine=fine,
            coarse_commit=grp['event_summary'][0]['commit_accepted'],
            fine_commit=grp['event_summary'][1]['commit_accepted'],
            fine_recross=grp['event_summary'][1]['recross_accepted'],
            accepted_paths=grp['accepted'],sampling='Stage5C-corrected repeat-consistent subset; N32/64 not uniformly PT-accurate'))
counts={}
for model in ('bethe','two_state','product'):
    for N in (32,64):
        sub=[x for x in rows if x['model']==model and x['N']==N]
        counts[f'{model}_N{N}']=dict(n=len(sub),structural_pass=sum(x['structural_pass'] for x in sub),
                                  fine_ARI_median=float(np.median([x['fine_ARI'] for x in sub])),
                                  overlap_gap_range=[float(min(x['overlap_gap'] for x in sub)),float(max(x['overlap_gap'] for x in sub))],
                                  times=[(x['half_coarse'],x['half_fine']) for x in sub],
                                  fine_commit_rate=float(np.mean([x['fine_commit']/(x.get('accepted_paths',24)) for x in sub])))
aud=[]
for index in range(4):aud.append(json.loads((R/f'BETHE_AUDIT_{index:02d}.json').read_text()))
summary=dict(stage='5E',rows=rows,aggregate=counts,
             frozen_random_bethe_audits=aud,
             controls_model='Product iid Rademacher; symmetric two-state mixture of iid spins with per-site mean 0.8 conditional on global latent sign',
             gamma_window='21 original correlated Gaussian observation times, 0 to 3.2',
             joint_gate='ARI coarse >=0.9; fine>=0.7; each leaf mass >=0.03; within-minus-sibling conditional overlap >=0.10',
             interpretive_limit='The six selected Bethe disorders had previously passed the structural gate; control seeds are independent training replicates rather than quenched disorders. No asymptotic or controlled population-level significance is inferred. PT audit checks only four randomly selected fields; two Stage5D unresolved paths remain unresolved.')
(R/'STAGE5E_RESULTS.json').write_text(json.dumps(summary,indent=2))
# Figure 1: joint gate not merely information ordering. The structural panels are drawn
# from the same threshold definitions; points not independently sampled across models.
fig,axs=plt.subplots(1,3,figsize=(13,3.7))
order=('bethe','two_state','product');markers=('o','s','^')
for i,model in enumerate(order):
    sub=[x for x in rows if x['model']==model]
    off=(i-1)*.15
    for n,shape in [(32,'filled'),(64,'hollow')]:
        z=[x for x in sub if x['N']==n]
        axs[0].scatter([i+off+(.035 if n==64 else -.035)]*len(z),[x['fine_ARI'] for x in z],s=45,marker=markers[i],edgecolor='black',facecolor='white' if n==64 else f'C{i}',zorder=4)
        axs[1].scatter([i+off+(.035 if n==64 else -.035)]*len(z),[x['overlap_gap'] for x in z],s=45,marker=markers[i],edgecolor='black',facecolor='white' if n==64 else f'C{i}',zorder=4)
        axs[2].scatter([x['half_coarse'] for x in z],[x['half_fine'] for x in z],s=48,marker=markers[i],edgecolor='black',facecolor='white' if n==64 else f'C{i}',label=f"{dict(bethe='Bethe',two_state='two-state',product='independent')[model]}, N={n}")
axs[0].axhline(.7,c='grey',ls='--',lw=1);axs[1].axhline(.1,c='grey',ls='--',lw=1)
for j in (0,1):axs[j].set_xticks(range(3),['Bethe glass','two-state\nmixture','independent\nspins']);axs[j].grid(axis='y',alpha=.2)
axs[0].set(title='(a) reproducibility of the first glassy split\non held-out samples',ylabel='adjusted Rand index',ylim=(-.07,1.06))
axs[1].set(title='(b) overlap within a class minus\noverlap with its sibling class',ylabel='overlap gap',ylim=(-.03,.47))
axs[2].plot([0,3.2],[0,3.2],ls=':',color='gray',lw=1)
axs[2].set(title='(c) decision times',xlabel='sector $\\gamma_{1/2}$',ylabel='first glassy level $\\gamma_{1/2}$',xlim=(-.05,2),ylim=(-.05,2))
axs[2].legend(fontsize=6,ncol=2,loc='lower right')
fig.tight_layout();fig.savefig(F/'E5E_joint_structural_dynamic.png',dpi=190);fig.savefig(F/'E5E_joint_structural_dynamic.pdf');plt.close(fig)
# Figure 2: randomly selected PT audit independent of sample quality.
fig,ax=plt.subplots(figsize=(7.5,3.5))
xx=np.arange(4);vals=[a['stage5d_corrected_vs_new_PT_TV'] for a in aud]
ax.bar(xx,vals);ax.axhline(.1,color='gray',ls='--',label='Stage5C repeatability threshold (not PT accuracy tolerance)')
ax.set(xticks=xx,xticklabels=[f"N={a['frozen_case']['N']}\nseed {a['frozen_case']['seed']}, path {a['frozen_case']['path']}" for a in aud],
       ylabel='Total-variation distance (four class masses)',title='Stage 5E: four preselected independent Bethe posterior checks',ylim=(0,.11))
ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(F/'E5E_random_audit.png',dpi=190);fig.savefig(F/'E5E_random_audit.pdf');plt.close(fig)
for k,v in counts.items(): print(k,v,flush=True)
print('auditTV',vals,'GATES',[a['pt_gate'] for a in aud])
returncode=0
