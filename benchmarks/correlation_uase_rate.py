"""Per-gene error of UASE on soft-powered and TOM networks vs number of samples (factor model, 1000 genes).

Checks the entrywise-perturbation argument in /mnt/project-files/reviews/correlation-uase.md: the largest
entry error of the network and the largest per-gene embedding error (aligned, at an eigengap) should
both shrink like sqrt(log p / n). Run with python benchmarks/correlation_uase_rate.py (a few minutes).
"""
import numpy as np, sys
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
from node2vec2rank.embedding import uase
from benchmarks.correlation_uase import loadings, sample, standardise, NUM_FACTORS
rng=np.random.default_rng(3)
def net(c, kind):
    if kind=='hd': a=((1+c)/2)**10
    elif kind=='abs': a=np.abs(c)**10
    elif kind=='tom':
        a=((1+c)/2)**10; np.fill_diagonal(a,0); k=a.sum(1); a=(a@a+a)/(np.minimum.outer(k,k)+1-a)
    np.fill_diagonal(a,0); return a
def align(Y,T):
    # Procrustes on the stacked embeddings
    u,_,vt=np.linalg.svd(Y.reshape(-1,Y.shape[-1]).T@T.reshape(-1,T.shape[-1])); return Y@ (u@vt)
p=1000; d=12
for design in ('mixed',):
  b,a,_,_=loadings(p,rng,design,'switch')
  for kind in ('hd','tom','abs'):
    pop=[net(b@b.T+np.diag(1-(b*b).sum(1)),kind), net(a@a.T+np.diag(1-(a*a).sum(1)),kind)]
    sv=np.linalg.svd(np.hstack(pop),compute_uv=False)[:40]
    d=int(np.argmax(sv[1:30]/sv[2:31]))+2  # largest relative gap
    print(kind,'d at gap',d,'ratio %.2f'%(sv[d-1]/sv[d]), 'sv',np.round(sv[:d+2],1))
    T=uase(pop,d,random_state=0); scale=np.linalg.norm(T,axis=2).max()
    for n in (50,150,500,2000,8000):
        errs=[];ents=[]
        for r in range(2):
            C=[np.corrcoef(sample(L,n,rng),rowvar=False) for L in (b,a)]
            G=[net(c,kind) for c in C]
            ents.append(max(np.abs(G[k]-pop[k]).max() for k in range(2)))
            Y=align(uase(G,d,random_state=0),T)
            errs.append(np.linalg.norm(Y-T,axis=2).max()/scale)
        print(kind,n,'max entry err %.4f'%np.mean(ents),'max per-gene err (rel) %.3f'%np.mean(errs),'sqrt(log p/n) %.3f'%np.sqrt(np.log(p)/n),flush=True)
