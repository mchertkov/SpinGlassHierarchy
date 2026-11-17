#!/usr/bin/env python3
"""Supplement for paper11_14_26. Run from any working directory.
Reconstruct original bethe disorder seeds from archived N1 result identifiers, use
unchanged exacttree.build_tree, calculate exact overlap residual and compare binary
channel predictions. No class gates or source dataset are modified.

Requires numpy, scipy, matplotlib, numba (same as original project).
Optional --limit-seeds argument for quick smoke tests.
"""
from __future__ import annotations
import os, sys, json, csv, argparse, math
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'experiments'))
import exacttree as E
import n1_levels as N1

hx,hw = hermgauss(72)
zpts=np.sqrt(2.0)*hx
zw=hw/np.sqrt(np.pi)

def hb(w):
    w=float(w)
    return -(w*np.log(w)+(1-w)*np.log1p(-w))

def binary_information(g, w):
    w=float(w)
    a=np.log(w/(1.-w)); dz=np.sqrt(max(g,0))*zpts
    l1=a+g/2+dz; l0=a-g/2+dz
    # H(C|L), evaluate via stable entropy of sigmoid logits
    def ent_logit(l):
        return np.logaddexp(0.,l)-l*expit(l)
    return max(0.,hb(w)-float(w*np.dot(zw,ent_logit(l1))+(1-w)*np.dot(zw,ent_logit(l0))))

def predicted_massaware(splits):
    # each accepted node is one conditional binary channel, weighted by parent mass
    rows=[]
    for x in splits:
        m=x['mass']; s1,s2=x['masses']; w=s1/(s1+s2)
        if not (0.00001<w<0.99999): continue
        kap=x['kappa']; d2=kap/(w*(1-w))
        rows.append((m,w,d2))
    if not rows:return np.nan
    total=sum(m*hb(w) for m,w,d2 in rows)
    def fraction(g):return sum(m*binary_information(g*d2,w) for m,w,d2 in rows)/total
    hi=0.1
    while fraction(hi)<.5 and hi<1e5:hi*=2
    return float(brentq(lambda g:fraction(g)-.5,0,hi,xtol=1e-8))

def residual_levels(M,T):
    C0=M.C0; target=float(np.sum(C0*C0))/M.N**2; ans=[]
    prev=float('inf')
    for lab,K in zip(T['labels'],T['n_classes']):
        masses=np.bincount(lab.ravel(),weights=M.p.ravel(),minlength=K)
        Mmat=np.zeros((M.N,M.N))
        for k in range(K):
            if masses[k]<1e-12:continue
            first=M.first_moment(np.where(lab==k,M.p,0.0))
            Mmat+=np.outer(first,first)/masses[k]
        pred=float(np.sum(Mmat*Mmat))/M.N**2
        r=max(0,target-pred)
        if r>prev+2e-8:raise AssertionError('R nonmonotonic under refinement')
        prev=r
        diag=float((M.N-np.sum(np.diag(Mmat)**2))/M.N**2)
        ans.append(dict(R=r,Rdiag=diag,Roff=r-diag,explained=pred/target if target>0 else np.nan,nclass=K,q2=target))
    return ans

def gen_rows(n=16,family='bethe',betas=(1.,1.5,2.5),limit=None):
    rec=[]; archive=ROOT/'results'/'N1'
    import re
    for beta in betas:
        tag=f'{family}_N{n}_b{beta}_s'
        items=[]
        for p in archive.glob(f'{tag}*.json'):
            if p.stem.endswith('_anatomy'):continue
            mm=re.search(r'_s(\d+)$',p.stem)
            if mm:items.append((int(mm.group(1)),p))
        items.sort()
        if limit is not None:items=items[:limit]
        for seed,p in items:
            saved=json.loads(p.read_text());M=E.ExactModel(N1.J_of(family,n,beta,seed))
            T=E.build_tree(M)
            if T['n_classes']!=saved['n_classes']:
                raise RuntimeError(f'NOT REPRODUCIBLE: {p.name}: {T["n_classes"]}!={saved["n_classes"]}')
            lv=residual_levels(M,T)
            d={'family':family,'N':n,'beta':beta,'seed':seed,'name':p.stem,'n_glassy':len(lv)-1}
            for ix,s in enumerate(lv):
                rec.append({**d,'depth':ix,**s,'initial':lv[0]['R'],'terminal':lv[-1]['R']})
            if len(rec)%50==0:print('completed residual rows',len(rec),flush=True)
    return rec

def gather_channel_rows():
    import re
    rows=[]
    for f in (ROOT/'results'/'N1').glob('*.json'):
        if 'anatomy' in f.stem:continue
        md=re.match(r'(bethe|sk)_N(16|20)_b([\d.]+)_s(\d+)$',f.stem)
        if md is None:continue
        family,n,beta,seed=md.groups()
        x=json.loads(f.read_text())
        for i,splits in enumerate(x.get('level_info',[]),start=1):
            if i>=len(x['half_gamma']): continue
            meas=x['half_gamma'][i]
            if meas is None or meas<=0:continue
            pred0=x['pred_cond'][i] if i<len(x['pred_cond']) else np.nan
            pm=predicted_massaware(splits)
            minor=min(min(s['masses'])/sum(s['masses']) for s in splits)
            rows.append(dict(family=family,N=int(n),beta=float(beta),seed=int(seed),level=i,measured=meas,spectral=pred0,mass_aware=pm,smallest_child=minor,n_split=len(splits),logerr_spectral=abs(np.log(pred0/meas)),logerr_mass=abs(np.log(pm/meas))))
    return rows

def save_csv(name,rows):
    f=OUTPUT/'extra_results'/name
    with open(f,'w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('wrote',f,'rows',len(rows),flush=True)

def figures(resid,channel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot only terminal original-algorithm partition vs sector-only for each size.
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.6),sharey=True)
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax,n in zip(axes,(16,20)):
        for ib,b in enumerate((1.,1.5,2.5)):
            r=[x for x in resid if x['N']==n and x['beta']==b]
            by={x['seed']:x for x in r if x['depth']==x['n_glassy']}
            v=np.array([x['terminal'] for x in by.values()]);u=np.array([x['initial'] for x in by.values()]);
            if not len(v):continue
            col=colors[ib]
            ax.scatter([b-.08]*len(v),u,marker='x',alpha=.25,s=17,color=col)
            ax.scatter([b+.08]*len(v),v,marker='o',alpha=.25,s=17,color=col)
            ax.errorbar([b-.08,b+.08],[u.mean(),v.mean()],yerr=[u.std(ddof=1)/len(u)**.5,v.std(ddof=1)/len(v)**.5],
                        marker='D',capsize=3,linestyle='-',color=col)
            ax.text(b,v.mean()+.009,f'{len(v)}',fontsize=7,ha='center')
        ax.set_title(f'{n} spins; original disorder seeds')
        ax.set_xticks([1.,1.5,2.5]);ax.set_xlabel(r'coupling $\beta_J$')
        ax.set_xlim(.72,2.75);ax.grid(alpha=.2)
    axes[0].set_ylabel(r'$R_{J,N}$ (left x: sectors; right o: terminal cells)')
    fig.text(.50,.005,'Markers: individual disorder instances; diamonds: disorder means ± SEM; labels: sample sizes',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]);base=OUTPUT/'figures'/'FigA5_overlap_residual'
    fig.savefig(str(base)+'.pdf',bbox_inches='tight');fig.savefig(str(base)+'.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('figure',base,flush=True)
    # Separate mass-aware prediction comparison to see if robust, no assertion of improvement.
    channel=[r for r in channel if np.isfinite(r['spectral']) and np.isfinite(r['mass_aware'])]
    fig,ax=plt.subplots(figsize=(6.7,4.2))
    x=np.array([z['smallest_child'] for z in channel]);ds=np.array([z['logerr_spectral'] for z in channel]);dm=np.array([z['logerr_mass'] for z in channel]);
    ax.scatter(x,ds,s=13,alpha=.35,label='Spectral approximation')
    ax.scatter(x,dm,s=13,alpha=.35,label='Mass-aware rigid-child channel')
    bins=[.0,.08,.16,.24,.34,.50]
    med_x=[]; med_spec=[]; med_mass=[]
    for a,b in zip(bins[:-1],bins[1:]):
        idx=(x>=a)&(x<b)
        if idx.sum()>=5:
            med_x.append((a+b)/2)
            med_spec.append(np.median(ds[idx]))
            med_mass.append(np.median(dm[idx]))
    ax.plot(med_x,med_spec,'D-',color=colors[0],lw=1.5,ms=6,label='Binned spectral median')
    ax.plot(med_x,med_mass,'s-',color=colors[1],lw=1.5,ms=6,label='Binned mass-aware median')
    ax.set_xlabel('Smallest conditional child mass at level');ax.set_ylabel(r'$|\log(\gamma_{\rm pred}/\gamma_{\rm obs})|$')
    ax.set_title('Original exact Ising decision data: two predictions');ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8)
    fig.tight_layout();base=OUTPUT/'figures'/'FigA6_mass_aware'
    fig.savefig(str(base)+'.pdf',bbox_inches='tight');fig.savefig(str(base)+'.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('figure',base,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit-seeds',type=int,default=None)
    a=ap.parse_args()
    residual=[]
    for n in (16,20):
        rr=gen_rows(n=n,limit=a.limit_seeds);residual+=rr
        print('Completed N',n, 'instances',len({(r['beta'],r['seed']) for r in rr}),flush=True)
    save_csv('exact_overlap_residual.csv',residual)
    channel=gather_channel_rows();save_csv('mass_aware_comparison.csv',channel)
    figures(residual,channel)
    last=[r for r in residual if r['depth']==r['n_glassy']]
    summ={}
    for n in (16,20):
        for b in (1.,1.5,2.5):
            r=[x for x in last if x['N']==n and x['beta']==b]
            if not r:continue
            v=np.array([x['terminal'] for x in r]);u=np.array([x['initial'] for x in r]);
            summ[f'{n}:{b}']={'n':len(r),'mean_initial':float(u.mean()),'mean_terminal':float(v.mean()),'se_terminal':float(v.std(ddof=1)/np.sqrt(len(v))),'fraction_with_glassy':float(np.mean([x['n_glassy']>0 for x in r]))}
    summ['channel']={'n':len(channel),'median_abs_logerr_spectral':float(np.median([r['logerr_spectral'] for r in channel])),'median_abs_logerr_mass':float(np.median([r['logerr_mass'] for r in channel])),'fraction_mass_better':float(np.mean([r['logerr_mass']<r['logerr_spectral'] for r in channel]))}
    (OUTPUT/'extra_results'/'summary.json').write_text(json.dumps(summ,indent=2))
    print(json.dumps(summ,indent=2),flush=True)
if __name__=='__main__':main()
