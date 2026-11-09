"""Compiled serial graph-color heat bath and PT; fully explicit, resumable PCG64 streams.

The state evolution and RNG stream are deterministic for a fixed sweep partition,
including the global Z2 proposal using an independent generator.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def simulate(X,E,H,tag,seen,trips,acc,att,flipatt,flipok,
             betas,colors,nbr,jn,sw0,n,burn,stride,flip_stride,do_flip,rng,rngflip):
    R,C,N=X.shape
    out=np.empty((n//stride+3,C,N),np.int8)
    outs=np.empty((n//stride+3,C),np.float64)
    saved=0
    for sw in range(sw0,sw0+n):
        # One heat-bath update per spin (each coloring class is an independent set).
        for color in range(colors.shape[0]):
            for r in range(R):
                for c in range(C):
                    for ci in range(colors.shape[1]):
                        i=colors[color,ci]
                        if i<0:continue
                        loc=0.0
                        for k in range(nbr.shape[1]):
                            if nbr[i,k]>=0 and nbr[i,k]<N:
                                loc+=jn[i,k]*X[r,c,nbr[i,k]]
                        z=betas[r]*loc+H[c,i]
                        # robust logistic / tanh representation
                        p=0.5*(1.0+np.tanh(z))
                        new=1 if rng.random()<p else -1
                        old=X[r,c,i]
                        if new!=old:
                            X[r,c,i]=new
                            E[r,c]+=(new-old)*loc
        if do_flip and flip_stride>0 and (sw+1)%flip_stride==0:
            for r in range(R):
                for c in range(C):
                    hx=0.0
                    for i in range(N): hx+=H[c,i]*X[r,c,i]
                    flipatt[r]+=1
                    if np.log(max(1e-300,rngflip.random())) < -2.0*hx:
                        for i in range(N):X[r,c,i]=-X[r,c,i]
                        flipok[r]+=1
        for r in range(sw%2,R-1,2):
            for c in range(C):
                att[r]+=1
                if np.log(max(1e-300,rng.random())) < (betas[r]-betas[r+1])*(E[r+1,c]-E[r,c]):
                    acc[r]+=1
                    for i in range(N):
                        tmp=X[r,c,i];X[r,c,i]=X[r+1,c,i];X[r+1,c,i]=tmp
                    ee=E[r,c];E[r,c]=E[r+1,c];E[r+1,c]=ee
                    tt=tag[r,c];tag[r,c]=tag[r+1,c];tag[r+1,c]=tt
        for c in range(C):
            pos=0
            for r in range(R):
                if tag[r,c]==0:pos=r;break
            if pos==R-1:seen[c]=True
            if pos==0 and seen[c]:
                trips[c]+=1;seen[c]=False
        if sw>=burn and (sw-burn)%stride==0:
            for c in range(C):
                outs[saved,c]=E[R-1,c]
                for i in range(N):out[saved,c,i]=X[R-1,c,i]
            saved+=1
    return out[:saved],outs[:saved]
