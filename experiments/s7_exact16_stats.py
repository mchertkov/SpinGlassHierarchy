import json, glob, numpy as np
from scipy.stats import spearmanr, mannwhitneyu
R=[json.load(open(f)) for f in sorted(glob.glob('results/S7/exact16_*_seed*.json'))]
rows=[]
for r in R:
    c,f=r['measured_half']; p1,p2=r['spectral'][:2]
    exr=np.array(r['detectors']['excess_rel']); ex=np.array(r['detectors']['excess'])
    rows.append(dict(name=r['file'][8:-5],kind=r['kind'],acc=r['accepted'],rc=c/p1,rf=f/p2,l2b=r['lam2_over_bulk'],g23=r['gap23'],g12=r['gap12'],Q2=r['Q2'],
             exrel=exr.max(),exabs=ex.max(),std=r['peaks']['std_lograte']/f,var=r['peaks']['var_chi']/c,c=c,f=f,p1=p1,p2=p2,ari=r['heldout_ARI'][1]))
print(f"{'instance':22s} acc  coarse/(1/l1) fine/(1/l2) l2/bulk gap12 gap23  Q2   max rel.excess  stdpk/fine")
for w in rows:
    print(f"{w['name']:22s} {str(w['acc'])[0]}   {w['rc']:6.2f}      {w['rf']:6.2f}     {w['l2b']:6.2f} {w['g12']:6.1f} {w['g23']:5.2f} {w['Q2']:.2f}   {w['exrel']:7.3f}       {w['std']:5.2f}")
B=[w for w in rows if w['kind']=='bethe']; A=[w for w in B if w['acc']]; NA=[w for w in B if not w['acc']]
print("Bethe Spearman(coarse half,1/l1)=",round(spearmanr([w['c'] for w in B],[w['p1'] for w in B]).correlation,2),"ratio median",round(np.median([w['rc'] for w in B]),2))
print("Spearman(fine half,1/l2): all Bethe",round(spearmanr([w['f'] for w in B],[w['p2'] for w in B]).correlation,2),"accepted",round(spearmanr([w['f'] for w in A],[w['p2'] for w in A]).correlation,2))
print("fine ratio accepted",np.round([w['rf'] for w in A],2),"median",round(np.median([w['rf'] for w in A]),2)," rejected",np.round([w['rf'] for w in NA],2))
for key in ['l2b','g12','g23','exrel']:
    a=[w[key] for w in A]; n=[w[key] for w in NA]; print(f"{key}: accepted median {np.median(a):.2f}, rejected median {np.median(n):.2f}, MWU p={mannwhitneyu(a,n).pvalue:.3f}")
for k in ['product','two_state']:
    S=[w for w in rows if w['kind']==k]; print(k,"l2/bulk",round(np.median([w['l2b'] for w in S]),2),"max rel excess",round(np.median([w['exrel'] for w in S]),3),"coarse ratio",round(np.median([w['rc'] for w in S]),2),"fine ratio",round(np.median([w['rf'] for w in S]),2))
print("Bethe max rel excess median",round(np.median([w['exrel'] for w in B]),3),"range",round(min(w['exrel'] for w in B),3),round(max(w['exrel'] for w in B),3))
json.dump(rows,open('results/S7/exact16_cohort_table.json','w'),indent=1)

# ---- interpolated half-information times and the binary-channel law gamma_half = s*/lambda (s* = 1.042)
S_STAR = 1.042
def interp_half(g, frac, lam=0.5):
    for k in range(1, len(g)):
        if frac[k - 1] < lam <= frac[k] and g[k - 1] > 0:
            f = (lam - frac[k - 1]) / (frac[k] - frac[k - 1]); return float(np.exp(np.log(g[k - 1]) + f * (np.log(g[k]) - np.log(g[k - 1]))))
        if frac[k - 1] < lam <= frac[k]: return float(g[k] * (lam / max(frac[k], 1e-9)))
    return float('nan')
out = []
for f in sorted(glob.glob('results/E5F/exact16/*.json')):
    d = json.load(open(f)); x = np.array(d['information']); g = np.array(d['gamma'])
    s7 = json.load(open('results/S7/exact16_' + f.split('/')[-1]))
    hc = interp_half(g, x[:, 0] / x[-1, 0]); hf = interp_half(g, x[:, 1] / x[-1, 1]); lam = np.array(s7['lam'])
    out.append(dict(name=f.split('/')[-1][:-5], kind=d['kind'], acc=bool(d['accepted_candidate_tree']), half_c=hc, half_f=hf,
                    pred_c=S_STAR / lam[0], pred_f=S_STAR / lam[1], ratio_c=hc * lam[0] / S_STAR, ratio_f=hf * lam[1] / S_STAR))
for o in out: print(f"{o['name']:24s} {o['kind']:9s} acc={str(o['acc'])[0]}  coarse {o['half_c']:.3f} vs {o['pred_c']:.3f} (x{o['ratio_c']:.2f})   fine {o['half_f']:.3f} vs {o['pred_f']:.3f} (x{o['ratio_f']:.2f})")
for k in ('bethe', 'product', 'two_state'):
    S = [o for o in out if o['kind'] == k]
    print(k, 'coarse ratio median', round(np.median([o['ratio_c'] for o in S]), 2), 'IQR', np.round(np.percentile([o['ratio_c'] for o in S], [25, 75]), 2),
          '| fine ratio median', round(np.median([o['ratio_f'] for o in S]), 2))
A = [o for o in out if o['kind'] == 'bethe' and o['acc']]; print('accepted Bethe fine ratio', np.round([o['ratio_f'] for o in A], 2), 'median', round(np.median([o['ratio_f'] for o in A]), 2))
json.dump(out, open('results/S7/exact16_binary_channel_law.json', 'w'), indent=1)
