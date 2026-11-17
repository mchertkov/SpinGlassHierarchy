"""Exact overlap residual for the N=24 Bethe figure-2 anatomy case (seed 2, beta 2.5)."""
import os,sys,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'code'),str(root/'experiments'),str(Path(__file__).parent)]
import exacttree as E,n1_levels as N1
from validate_cells_and_channels import residual_levels
start=time.time(); M=E.ExactModel(N1.J_of('bethe',24,2.5,2));print('model',time.time()-start,flush=True)
T=E.build_tree(M);print('tree',time.time()-start, T['n_classes'],flush=True)
archive=json.loads((root/'results'/'N1'/'bethe_N24_b2.5_s2_anatomy.json').read_text())
assert T['n_classes']==archive['n_classes'], (T['n_classes'],archive['n_classes'])
vals=residual_levels(M,T)
out={'N':24,'beta':2.5,'seed':2,'nclasses':T['n_classes'],'levels':vals,'elapsed_seconds':time.time()-start,'note':'Single prespecified anatomy instance; not a disorder average'}
(root/'paper11_14_26'/'extra_results'/'n24_anatomy_overlap.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)

# The depth series is a single prespecified case, visually distinct from cohort means.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6.2,3.15))
depth=list(range(len(vals)))
ax.plot(depth,[v['R'] for v in vals],'-o',label='Full conditional overlap residual')
ax.plot(depth,[v['Roff'] for v in vals],'--s',label='Off-diagonal component')
ax.plot(depth,[v['Rdiag'] for v in vals],':^',label='Diagonal component')
ax.set_xticks(depth,[f'{i}\n{vals[i]["nclass"]} cells' for i in depth])
ax.set_ylabel(r'Exact $R_{J,N}$')
ax.set_xlabel('Glassy hierarchy depth (sector = 0)')
ax.set_title(r'Prespecified $N=24$, $\beta_J=2.5$ anatomy instance')
ax.grid(alpha=.2)
ax.legend(fontsize=8)
fig.tight_layout()
base=root/'paper11_14_26'/'figures'/'FigA7_anatomy_overlap'
fig.savefig(str(base)+'.pdf',bbox_inches='tight')
fig.savefig(str(base)+'.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print('created',base,flush=True)
