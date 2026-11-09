"""Preselected, bounded new Stage5E field/time PT audit; no replacement runs."""
from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'code')]
from e5c_fieldwise_audit import one_pt
from e5d_analyze import reconstruct
from e5_largeN_importance import load
from e3b_e1_heldout_hierarchy import train_tree
outdir=ROOT/'results'/'E5E';outdir.mkdir(exist_ok=True,parents=True)


def one(index):
    cases=json.loads((outdir/'FROZEN_PROTOCOL.json').read_text())['posterior_PT_random_audit']
    c=cases[index];N=c['N'];seed=c['seed'];path=c['path'];g=float(c['gamma'])
    zz=np.load(ROOT/'results'/'E5B'/f'N{N}_bethe_seed{seed}_p24_P768.npz')
    idx=int(np.where(np.isclose(zz['gamma'],g))[0][0]);h=zz['H'][path,idx]
    gf,Y,m,err,repaired,unresolved,anch,ptgate,orig=reconstruct(N,seed)
    assert np.array_equal(gf,zz['gamma']) and np.array_equal(Y,zz['Y'])
    a,_=load(N,seed,0)
    tree=train_tree(np.r_[a[:,:8].reshape(-1,N),-a[:,:8].reshape(-1,N)])
    t=time.monotonic()
    ref=one_pt(N,seed,h,tree,seed=790010+index*9971,sweeps=16000,stride=20)
    mass=ref['mass'];est=m[path,idx]
    original=orig[path,:,idx]
    doc=dict(stage='5E',kind='random_preselected_field_PT',frozen_case=c,index=index,
             field_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
             pt_sweeps=16000,pt_gate=bool(ref['gate']),
             pt_roundtrips=[int(v) for v in ref['roundtrips']],
             pt_start_TV=float(ref['start_tv']),pt_split_TV=float(ref['split_tv']),
             pt_energy_split_per_spin=float(ref['energy_split_per_spin']),
             independently_generated_PT_leaf_mass=mass.tolist(),
             stage5d_corrected_leaf_mass=est.tolist(),
             stage5d_corrected_vs_new_PT_TV=float(.5*abs(mass-est).sum()),
             original_two_pop_vs_PT_TV=[float(.5*abs(mass-x).sum()) for x in original],
             stage5c_repaired_path=bool(repaired[path]),stage5c_unresolved_path=bool(unresolved[path]),
             elapsed_s=float(time.monotonic()-t),
             limitation='Single independent PT reference with operational gate, not exact-equilibrium certification; no retries; fields were selected uniformly before audit.')
    prefix=outdir/f'BETHE_AUDIT_{index:02d}'
    prefix.with_suffix('.json').write_text(json.dumps(doc,indent=2))
    np.savez_compressed(prefix.with_suffix('.npz'),h=h,pt_mass=mass,pt_energies=ref['energies'],pt_leaf_samples=ref['leaf'])
    print('E5E AUDIT',index,c,'pass',doc['pt_gate'],'tv',round(doc['stage5d_corrected_vs_new_PT_TV'],3),'tripsmin',min(doc['pt_roundtrips']),'sec',round(doc['elapsed_s'],1),flush=True)
    return doc
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    one(a.index)
